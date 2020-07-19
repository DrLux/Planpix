from env import ControlSuiteEnv,EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
import numpy as np
import torch
from torch import nn, optim
import os
from planner import MPCPlanner
import random
from torch.distributions import Normal
from math import inf
from tqdm import tqdm
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import lineplot, write_video
from torchvision.utils import make_grid, save_image






### ENV ####
env_name = 'reacher-easy'
seed = 1
max_episode_length = 1000
bit_depth = 5
######

### Experience Replay
ex_replay_buff_size = 1000000
seed_episodes = 4 #primo episodio random per inizializzare il buffer
###

# Setup
results_dir = os.path.join('/home/luca/Desktop/luca/agosto/experience/')
os.makedirs(results_dir, exist_ok=True) #if flag is False then, if the target directory already exists an OSError is raised

os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available(): #and not args.disable_cuda:
  device = torch.device('cuda')
  torch.cuda.manual_seed(seed)
else:
  device = torch.device('cpu')

metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}
env = ControlSuiteEnv(env_name, seed, max_episode_length, bit_depth)

if os.path.exists(results_dir+'experience.pth'):  
    print("carico il buffer")
    D = torch.load(results_dir+'experience.pth')
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
    print("Total steps: ", D.steps)
    #print(metrics)
else:
    print("il buffer non esiste, lo creo")
    D = ExperienceReplay(ex_replay_buff_size, env.observation_size, env.action_size, bit_depth, device)

    for s in range(1, seed_episodes + 1):
        print("Episode: ", s)
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * env.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)

    # store experience to file
    #torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


#############################################

belief_size = 200
state_size = 30
hidden_size = 200
embedding_size = 1024
activation_function = 'relu'
learning_rate_schedule = 0 #Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)' 
learning_rate = 1e-3
adam_epsilon = 1e-4

# Initialise model parameters randomly
transition_model = TransitionModel(belief_size, state_size, env.action_size, hidden_size, embedding_size, activation_function).to(device=device)
observation_model = ObservationModel(belief_size, state_size, embedding_size, activation_function).to(device=device)
reward_model = RewardModel(belief_size, state_size, hidden_size, activation_function).to(device=device)
encoder = Encoder(embedding_size, activation_function).to(device=device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
optimiser = optim.Adam(param_list, lr=0 if learning_rate_schedule != 0 else learning_rate, eps=adam_epsilon)

#########################
planning_horizon = 12
optimisation_iters = 10
candidates = 1000
top_candidates = 100

planner = MPCPlanner(env.action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, env.action_range[0], env.action_range[1])

##################
batch_size = 50
free_nats = 3
action_noise = 0.3

global_prior = Normal(torch.zeros(batch_size, state_size, device=device), torch.ones(batch_size, state_size, device=device))  # Global prior N(0, I)
free_nats = torch.full((1, ), free_nats, dtype=torch.float32, device=device)  # Allowed deviation in KL divergence


def update_belief_and_act(env, planner, transition_model, encoder, belief, posterior_state, action, observation, min_action=-inf, max_action=inf, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = action + action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    action.clamp_(min=min_action, max=max_action)  # Clip action range
    next_observation, reward, done = env.step(action[0].cpu())  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done


#################
test_episodes = 3
flag_render = False
max_episode_length = 1000



#####################
episodes = 500
collect_interval = 100
chunk_size = 50
grad_clip_norm = 1000
test_interval = 25
checkpoint_interval  = 25

##### Carica i checkpoint del modello
model_dicts = torch.load(results_dir+'models_50.pth')
print("ho caricato il modello")
transition_model.load_state_dict(model_dicts['transition_model'])
observation_model.load_state_dict(model_dicts['observation_model'])
reward_model.load_state_dict(model_dicts['reward_model'])
encoder.load_state_dict(model_dicts['encoder'])
optimiser.load_state_dict(model_dicts['optimiser'])



# da (num_episodi_per_inizializzare) a (total_episodes + episodi_per_inizializzare)
for episode in tqdm(range(metrics['episodes'][-1] + 1, episodes + 1), total=episodes, initial=metrics['episodes'][-1] + 1):
    # Model fitting
    losses = []
    for s in tqdm(range(collect_interval)):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        observations, actions, rewards, nonterminals = D.sample(batch_size, chunk_size)  # Transitions start at time t = 0
        # Create initial belief and state for time t = 0
        init_belief, init_state = torch.zeros(batch_size, belief_size, device=device), torch.zeros(batch_size, state_size, device=device)
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:], )), nonterminals[:-1])
        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        # LOSS
        observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum((2, 3, 4)).mean(dim=(0, 1))
        reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))
        kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out

        # Update model parameters
        optimiser.zero_grad()
        (observation_loss + reward_loss + kl_loss).backward()
        nn.utils.clip_grad_norm_(param_list, grad_clip_norm, norm_type=2)
        optimiser.step()
        # Store (0) observation loss (1) reward loss (2) KL loss
        losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])

    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
    
    # Data collection
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action = torch.zeros(1, belief_size, device=device), torch.zeros(1, state_size, device=device), torch.zeros(1, env.action_size, device=device)
        pbar = tqdm(range(max_episode_length // env.action_repeat))
        for t in pbar:
            # QUI INVECE ESPLORI
            belief, posterior_state, action, next_observation, reward, done = update_belief_and_act(env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=device), env.action_range[0], env.action_range[1], explore=True)
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if flag_render:
                env.render()
            if done:
                pbar.close()
                break
        
    # Update and plot train reward metrics
    metrics['steps'].append(t + metrics['steps'][-1])
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)

    print('observation_loss -> from ', losses[0][0], "to ", losses[0][-1])
    print('reward_loss: -> from ', losses[1][0], "to ", losses[1][-1])
    print('kl_loss: -> from ', losses[2][0], "to ", losses[2][-1])
    print("metrics['train_rewards']: ", metrics['train_rewards'][-1])

    '''
    # Testing only -> NO MORE TESTING
    if episode % test_episodes == 0:
        print("Sto testando")
        # Set models to eval mode
        transition_model.eval()
        reward_model.eval()
        encoder.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(ControlSuiteEnv, (env_name, seed, max_episode_length, bit_depth), {}, test_episodes)

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((test_episodes, )), []
            belief, posterior_state, action = torch.zeros(test_episodes, belief_size, device=device), torch.zeros(test_episodes, state_size, device=device), torch.zeros(test_episodes, env.action_size, device=device)
            pbar = tqdm(range(max_episode_length // env.action_repeat))
            for t in pbar:
                # di default non fa exploration qui
                belief, posterior_state, action, observation, reward, done = update_belief_and_act(test_envs, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=device), env.action_range[0], env.action_range[1])
                total_reward += reward
                #video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
                observation = next_observation
                if done.sum().item() == test_episodes:
                    pbar.close()
                    print('Average Reward:', total_reward / test_episodes)
                    break

        # Update and plot reward metrics (and write video if applicable) and save metrics
        metrics['test_episodes'].append(episode)
        metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
        episode_str = str(episode).zfill(len(str(args.episodes)))
        write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
        save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        # Close test environments
        test_envs.close()
    '''
    '''
    ##test model
    if episode % test_episodes == 0:
        print("sto testando il modello")
        # Set models to eval mode
        transition_model.eval()
        reward_model.eval()
        encoder.eval()
        with torch.no_grad():
            total_reward = 0
            video_frames = []
            for _ in tqdm(range(test_episodes)):
                observation = env.reset()
                belief, posterior_state, action = torch.zeros(1, belief_size, device=device), torch.zeros(1, state_size, device=device), torch.zeros(1, env.action_size, device=device)
                pbar = tqdm(range(max_episode_length // env.action_repeat))
                video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
                for t in pbar:
                    belief, posterior_state, action, observation, reward, done = update_belief_and_act(env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=device), env.action_range[0], env.action_range[1])
                    total_reward += reward
                    if done:
                        print("sto creando il video")
                        write_video(video_frames, 'test_episode_%s' % str(episode), results_dir)  # Lossy compression
                        pbar.close()
                        break
        print('Average Reward:', total_reward / test_episodes)
    '''
        

    # Set models to train mode
    transition_model.train()
    observation_model.train()
    reward_model.train()
    encoder.train()
    
    '''
    if episode % checkpoint_interval == 0:
        torch.save({'transition_model': transition_model.state_dict(), 'observation_model': observation_model.state_dict(), 'reward_model': reward_model.state_dict(), 'encoder': encoder.state_dict(), 'optimiser': optimiser.state_dict()}, os.path.join(results_dir, 'models_%d.pth' % episode))
    '''
env.close()
quit()

