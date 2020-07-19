from math import inf
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
import numpy as np
from torch import nn, optim
from planner import MPCPlanner
import random
from torch.distributions import Normal
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import lineplot, write_video
from torchvision.utils import make_grid, save_image
import torch
import os
from env import EnvBatcher,ControlSuiteEnv
from tqdm import tqdm


class Trainer():
    def __init__(self, params, experience_replay_buffer,metrics,results_dir,env):
        self.parms = params     
        self.D = experience_replay_buffer  
        self.metrics = metrics
        self.env = env
        self.tested_episodes = 0
        self.results_dir = results_dir 
        self.model_path = self.results_dir # os.path.join(self.results_dir, 'model') 
        self.video_path = self.results_dir #os.path.join(self.results_dir, 'video') 

        # Create models
        self.transition_model = TransitionModel(self.parms.belief_size, self.parms.state_size, self.env.action_size, self.parms.hidden_size, self.parms.embedding_size, self.parms.activation_function).to(device=self.parms.device)
        self.observation_model = ObservationModel(self.parms.belief_size, self.parms.state_size, self.parms.embedding_size, self.parms.activation_function).to(device=self.parms.device)
        self.reward_model = RewardModel(self.parms.belief_size, self.parms.state_size, self.parms.hidden_size, self.parms.activation_function).to(device=self.parms.device)
        self.encoder = Encoder(self.parms.embedding_size,self.parms.activation_function).to(device=self.parms.device)
        self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(self.reward_model.parameters()) + list(self.encoder.parameters())
        self.optimiser = optim.Adam(self.param_list, lr=0 if self.parms.learning_rate_schedule != 0 else self.parms.learning_rate, eps=self.parms.adam_epsilon)
        self.planner = MPCPlanner(self.env.action_size, self.parms.planning_horizon, self.parms.optimisation_iters, self.parms.candidates, self.parms.top_candidates, self.transition_model, self.reward_model, self.env.action_range[0], self.env.action_range[1])
        
        global_prior = Normal(torch.zeros(self.parms.batch_size, self.parms.state_size, device=self.parms.device), torch.ones(self.parms.batch_size, self.parms.state_size, device=self.parms.device))  # Global prior N(0, I)
        self.free_nats = torch.full((1, ), self.parms.free_nats, dtype=torch.float32, device=self.parms.device)  # Allowed deviation in KL divergence

    def load_checkpoints(self):
        model_path = self.results_dir+'/best_model'
        os.makedirs(model_path, exist_ok=True) 
        files = os.listdir(model_path)
        if files:
            checkpoint = [f for f in files if os.path.isfile(os.path.join(model_path, f))]
            model_dicts = torch.load(os.path.join(model_path, checkpoint[0]))
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.observation_model.load_state_dict(model_dicts['observation_model'])
            self.reward_model.load_state_dict(model_dicts['reward_model'])
            self.encoder.load_state_dict(model_dicts['encoder'])
            self.optimiser.load_state_dict(model_dicts['optimiser'])  
            print("Loading models checkpoints!")
        else:
            print("Checkpoints not found!")
    
    def update_belief_and_act(self, env, belief, posterior_state, action, observation, min_action=-inf, max_action=inf, explore=False):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, self.encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)      
        if explore:
            action = action + self.parms.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        action.clamp_(min=min_action, max=max_action)  # Clip action range
        next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # If single env is istanceted perform single action (get item from list), else perform all actions
        return belief, posterior_state, action, next_observation, reward, done
    
    def fit_buffer(self,episode):
        ####
        # Prima fai uno step di training campionando dal dataset
        ######

        # Model fitting
        #losses = []
        tqdm.write("Fitting buffer")
        for s in tqdm(range(self.parms.collect_interval)):

            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
            observations, actions, rewards, nonterminals = self.D.sample(self.parms.batch_size, self.parms.chunk_size)  # Transitions start at time t = 0
            # Create initial belief and state for time t = 0
            init_belief, init_state = torch.zeros(self.parms.batch_size, self.parms.belief_size, device=self.parms.device), torch.zeros(self.parms.batch_size, self.parms.state_size, device=self.parms.device)
            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, bottle(self.encoder, (observations[1:], )), nonterminals[:-1])
            # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
            # LOSS
            observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum((2, 3, 4)).mean(dim=(0, 1))
            reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))
            kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), self.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out

            # Update model parameters
            self.optimiser.zero_grad()
            (observation_loss + reward_loss + kl_loss).backward() # BACKPROPAGATION
            nn.utils.clip_grad_norm_(self.param_list, self.parms.grad_clip_norm, norm_type=2)
            self.optimiser.step()
            # Store (0) observation loss (1) reward loss (2) KL loss
            #self.metrics['observation_loss'].append(observation_loss.item() )
            #self.metrics['reward_loss'].append(reward_loss.item() ) 
            #self.metrics['kl_loss'].append(kl_loss.item() )

        #save statistics and plot them
        losses = tuple(zip(*losses))  #PROVA A RIFARLO SENZA LOSSES
        self.metrics['observation_loss'].append(losses[0])
        self.metrics['reward_loss'].append(losses[1])
        self.metrics['kl_loss'].append(losses[2])
        lineplot(self.metrics['episodes'][-len(self.metrics['observation_loss']):], self.metrics['observation_loss'], 'observation_loss', self.results_dir)
        lineplot(self.metrics['episodes'][-len(self.metrics['reward_loss']):], self.metrics['reward_loss'], 'reward_loss', self.results_dir)
        lineplot(self.metrics['episodes'][-len(self.metrics['kl_loss']):], self.metrics['kl_loss'], 'kl_loss', self.results_dir)
        
    def explore_and_collect(self,episode):
        tqdm.write("Collect new data:")
        # Data collection
        with torch.no_grad():
            done = False
            observation, total_reward = self.env.reset(), 0
            belief, posterior_state, action = torch.zeros(1, self.parms.belief_size, device=self.parms.device), torch.zeros(1, self.parms.state_size, device=self.parms.device), torch.zeros(1, self.env.action_size, device=self.parms.device)
            t = 0
            for t in tqdm(range(self.parms.max_episode_length // self.env.action_repeat)):
                # QUI INVECE ESPLORI
                belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.parms.device), self.env.action_range[0], self.env.action_range[1], explore=True)
                self.D.append(observation, action.cpu(), reward, done)
                total_reward += reward
                observation = next_observation
                if self.parms.flag_render:
                    env.render()
                if done:
                    break

        # Update and plot train reward metrics
        self.metrics['steps'].append( (t * self.env.action_repeat) + self.metrics['steps'][-1])
        self.metrics['episodes'].append(episode)
        self.metrics['train_rewards'].append(total_reward)
        lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.results_dir)



    def train_models(self):
        # da (num_episodi_per_inizializzare) a (training_episodes + episodi_per_inizializzare)
        tqdm.write("Start training.")
        for episode in tqdm(range(self.parms.num_init_episodes +1, self.parms.num_init_episodes + self.parms.training_episodes) ):
            self.fit_buffer(episode)       
            self.explore_and_collect(episode)
            if episode % self.parms.test_interval == 0:
                self.test_model(episode)
                torch.save(self.metrics, os.path.join(self.results_dir, 'metrics.pth'))
                torch.save({'transition_model': self.transition_model.state_dict(), 'observation_model': self.observation_model.state_dict(), 'reward_model': self.reward_model.state_dict(), 'encoder': self.encoder.state_dict(), 'optimiser': self.optimiser.state_dict()}, os.path.join(self.model_path, 'models_%d.pth' % episode))

        return self.metrics

    def test_model(self, episode=None): #no explore here
        if episode is None:
            episode = self.tested_episodes
        # Set models to eval mode
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
        # Initialise parallelised test environments
        test_envs = EnvBatcher(ControlSuiteEnv, (self.parms.env_name, self.parms.seed, self.parms.max_episode_length, self.parms.bit_depth), {}, self.parms.test_episodes)

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((self.parms.test_episodes, )), []
            belief, posterior_state, action = torch.zeros(self.parms.test_episodes, self.parms.belief_size, device=self.parms.device), torch.zeros(self.parms.test_episodes, self.parms.state_size, device=self.parms.device), torch.zeros(self.parms.test_episodes, self.env.action_size, device=self.parms.device)
            tqdm.write("Testing model.")
            for t in range(self.parms.max_episode_length // test_envs.action_repeat): #floor division
                belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(test_envs,  belief, posterior_state, action, observation.to(device=self.parms.device), self.env.action_range[0], self.env.action_range[1])
                total_rewards += reward.numpy()
                # Collect real vs. predicted frames for video
                #observation_model(belief, posterior_state).cpu() ->  torch.from_numpy(prova).float
                #save_image(torch.as_tensor(frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % t))
                video_frames.append(make_grid(torch.cat([observation, self.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
                observation = next_observation
                if done.sum().item() == self.parms.test_episodes:
                    break
            
        #save and plot metrics 
        self.tested_episodes += 1
        self.metrics['test_episodes'].append(episode)
        self.metrics['test_rewards'].append(total_rewards.tolist())
        lineplot(self.metrics['test_episodes'], self.metrics['test_rewards'], 'test_rewards', self.results_dir)
        
        write_video(video_frames, 'test_episode_%s' % str(episode), self.video_path)  # Lossy compression
        # Set models to train mode
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
        # Close test environments
        test_envs.close()
        return self.metrics

