from math import inf
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
import numpy as np
from torch import nn, optim
from planner import MPCPlanner
import random
from torch.distributions import Normal
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import lineplot, write_video, double_lineplot
from torchvision.utils import make_grid, save_image
import torch
import os
from env import EnvBatcher,ControlSuiteEnv
from tqdm import tqdm
import cv2



class Trainer():
    def __init__(self, params, experience_replay_buffer,metrics,results_dir,env):
        self.parms = params     
        self.D = experience_replay_buffer  
        self.metrics = metrics
        self.env = env
        self.tested_episodes = 0

        self.statistics_path = results_dir+'/statistics' 
        self.model_path = results_dir+'/model' 
        self.video_path = results_dir+'/video' 
        self.rew_vs_pred_rew_path = results_dir+'/rew_vs_pred_rew'
        self.dump_plan_path = results_dir+'/dump_plan'
        
        #if folder do not exists, create it
        os.makedirs(self.statistics_path, exist_ok=True) 
        os.makedirs(self.model_path, exist_ok=True) 
        os.makedirs(self.video_path, exist_ok=True) 
        os.makedirs(self.rew_vs_pred_rew_path, exist_ok=True) 
        os.makedirs(self.dump_plan_path, exist_ok=True) 
        

        # Create models
        self.transition_model = TransitionModel(self.parms.belief_size, self.parms.state_size, self.env.action_size, self.parms.hidden_size, self.parms.embedding_size, self.parms.activation_function).to(device=self.parms.device)
        self.observation_model = ObservationModel(self.parms.belief_size, self.parms.state_size, self.parms.embedding_size, self.parms.activation_function).to(device=self.parms.device)
        self.reward_model = RewardModel(self.parms.belief_size, self.parms.state_size, self.parms.hidden_size, self.parms.activation_function).to(device=self.parms.device)
        self.encoder = Encoder(self.parms.embedding_size,self.parms.activation_function).to(device=self.parms.device)
        self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(self.reward_model.parameters()) + list(self.encoder.parameters()) 
        self.optimiser = optim.Adam(self.param_list, lr=0 if self.parms.learning_rate_schedule != 0 else self.parms.learning_rate, eps=self.parms.adam_epsilon)
        self.planner = MPCPlanner(self.env.action_size, self.parms.planning_horizon, self.parms.optimisation_iters, self.parms.candidates, self.parms.top_candidates, self.transition_model, self.reward_model,self.env.action_range[0], self.env.action_range[1])

        global_prior = Normal(torch.zeros(self.parms.batch_size, self.parms.state_size, device=self.parms.device), torch.ones(self.parms.batch_size, self.parms.state_size, device=self.parms.device))  # Global prior N(0, I)
        self.free_nats = torch.full((1, ), self.parms.free_nats, dtype=torch.float32, device=self.parms.device)  # Allowed deviation in KL divergence

    def load_checkpoints(self):
        self.metrics = torch.load(self.model_path+'/metrics.pth')
        model_path = self.model_path+'/best_model'
        os.makedirs(model_path, exist_ok=True) 
        files = os.listdir(model_path)
        if files:
            checkpoint = [f for f in files if os.path.isfile(os.path.join(model_path, f))]
            model_dicts = torch.load(os.path.join(model_path, checkpoint[0]),map_location=self.parms.device)
            self.transition_model.load_state_dict(model_dicts['transition_model'])
            self.observation_model.load_state_dict(model_dicts['observation_model'])
            self.reward_model.load_state_dict(model_dicts['reward_model'])
            self.encoder.load_state_dict(model_dicts['encoder'])
            self.optimiser.load_state_dict(model_dicts['optimiser'])  
            print("Loading models checkpoints!")
        else:
            print("Checkpoints not found!")


    def update_belief_and_act(self, env, belief, posterior_state, action, observation, reward, min_action=-inf, max_action=inf,explore=False):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        encoded_obs = self.encoder(observation).unsqueeze(dim=0).to(device=self.parms.device)       
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoded_obs)  # Action and observation need extra time dimension
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        action,pred_next_rew,_,_,_ = self.planner(belief, posterior_state,explore)  # Get action from planner(q(s_t|o≤t,a<t), p)      
        
        if explore:
            action = action + self.parms.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        action.clamp_(min=min_action, max=max_action)  # Clip action range
        next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # If single env is istanceted perform single action (get item from list), else perform all actions
        
        return belief, posterior_state, action, next_observation, reward, done,pred_next_rew 
    
    def fit_buffer(self,episode):
        ####
        # Fit data taken from buffer 
        ######

        # Model fitting
        losses = []
        tqdm.write("Fitting buffer")
        for s in tqdm(range(self.parms.collect_interval)):

            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
            observations, actions, rewards, nonterminals = self.D.sample(self.parms.batch_size, self.parms.chunk_size)  # Transitions start at time t = 0
            # Create initial belief and state for time t = 0
            init_belief, init_state = torch.zeros(self.parms.batch_size, self.parms.belief_size, device=self.parms.device), torch.zeros(self.parms.batch_size, self.parms.state_size, device=self.parms.device)
            encoded_obs = bottle(self.encoder, (observations[1:], ))

            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, encoded_obs, nonterminals[:-1])
            
            # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
            # LOSS
            observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum((2, 3, 4)).mean(dim=(0, 1))
            kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), self.free_nats).mean(dim=(0, 1))  
            reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))            

            # Update model parameters
            self.optimiser.zero_grad()

            (observation_loss + reward_loss + kl_loss).backward() # BACKPROPAGATION
            nn.utils.clip_grad_norm_(self.param_list, self.parms.grad_clip_norm, norm_type=2)
            self.optimiser.step()
            # Store (0) observation loss (1) reward loss (2) KL loss
            losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])#, regularizer_loss.item()])

        #save statistics and plot them
        losses = tuple(zip(*losses))  
        self.metrics['observation_loss'].append(losses[0])
        self.metrics['reward_loss'].append(losses[1])
        self.metrics['kl_loss'].append(losses[2])
      
        lineplot(self.metrics['episodes'][-len(self.metrics['observation_loss']):], self.metrics['observation_loss'], 'observation_loss', self.statistics_path)
        lineplot(self.metrics['episodes'][-len(self.metrics['reward_loss']):], self.metrics['reward_loss'], 'reward_loss', self.statistics_path)
        lineplot(self.metrics['episodes'][-len(self.metrics['kl_loss']):], self.metrics['kl_loss'], 'kl_loss', self.statistics_path)
        
    def explore_and_collect(self,episode):
        tqdm.write("Collect new data:")
        reward = 0
        # Data collection
        with torch.no_grad():
            done = False
            observation, total_reward = self.env.reset(), 0
            belief, posterior_state, action = torch.zeros(1, self.parms.belief_size, device=self.parms.device), torch.zeros(1, self.parms.state_size, device=self.parms.device), torch.zeros(1, self.env.action_size, device=self.parms.device)
            t = 0
            real_rew = []
            predicted_rew = [] 
            total_steps = self.parms.max_episode_length // self.env.action_repeat
            explore = True

            for t in tqdm(range(total_steps)):
                # Here we need to explore
                belief, posterior_state, action, next_observation, reward, done, pred_next_rew = self.update_belief_and_act(self.env, belief, posterior_state, action, observation.to(device=self.parms.device), [reward], self.env.action_range[0], self.env.action_range[1], explore=explore)
                self.D.append(observation, action.cpu(), reward, done)
                real_rew.append(reward)
                predicted_rew.append(pred_next_rew.to(device=self.parms.device).item())
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
        self.metrics['predicted_rewards'].append(np.array(predicted_rew).sum())

        lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.statistics_path)
        double_lineplot(self.metrics['episodes'], self.metrics['train_rewards'], self.metrics['predicted_rewards'], "train_r_vs_pr", self.statistics_path)

    def train_models(self):
        # from (init_episodes) to (training_episodes + init_episodes)
        tqdm.write("Start training.")

        for episode in tqdm(range(self.parms.num_init_episodes +1, self.parms.training_episodes) ):
            self.fit_buffer(episode)       
            self.explore_and_collect(episode)
            if episode % self.parms.test_interval == 0:
                self.test_model(episode)
                torch.save(self.metrics, os.path.join(self.model_path, 'metrics.pth'))
                torch.save({'transition_model': self.transition_model.state_dict(), 'observation_model': self.observation_model.state_dict(), 'reward_model': self.reward_model.state_dict(), 'encoder': self.encoder.state_dict(), 'optimiser': self.optimiser.state_dict()},  os.path.join(self.model_path, 'models_%d.pth' % episode))
            
            if episode % self.parms.storing_dataset_interval == 0:
                self.D.store_dataset(self.parms.dataset_path+'dump_dataset')

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
        total_steps = self.parms.max_episode_length // test_envs.action_repeat
        rewards = np.zeros(self.parms.test_episodes)
        
        real_rew = torch.zeros([total_steps,self.parms.test_episodes])
        predicted_rew = torch.zeros([total_steps,self.parms.test_episodes])

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((self.parms.test_episodes, )), []            
            belief, posterior_state, action = torch.zeros(self.parms.test_episodes, self.parms.belief_size, device=self.parms.device), torch.zeros(self.parms.test_episodes, self.parms.state_size, device=self.parms.device), torch.zeros(self.parms.test_episodes, self.env.action_size, device=self.parms.device)
            tqdm.write("Testing model.")
            for t in range(total_steps):     
                belief, posterior_state, action, next_observation, rewards, done, pred_next_rew  = self.update_belief_and_act(test_envs,  belief, posterior_state, action, observation.to(device=self.parms.device), list(rewards), self.env.action_range[0], self.env.action_range[1])
                total_rewards += rewards.numpy()
                real_rew[t] = rewards
                predicted_rew[t]  = pred_next_rew

                observation = self.env.get_original_frame().unsqueeze(dim=0)

                video_frames.append(make_grid(torch.cat([observation, self.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
                observation = next_observation
                if done.sum().item() == self.parms.test_episodes:
                    break
            
        real_rew = torch.transpose(real_rew, 0, 1)
        predicted_rew = torch.transpose(predicted_rew, 0, 1)
        
        #save and plot metrics 
        self.tested_episodes += 1
        self.metrics['test_episodes'].append(episode)
        self.metrics['test_rewards'].append(total_rewards.tolist())

        lineplot(self.metrics['test_episodes'], self.metrics['test_rewards'], 'test_rewards', self.statistics_path)
        
        write_video(video_frames, 'test_episode_%s' % str(episode), self.video_path)  # Lossy compression
        # Set models to train mode
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
        # Close test environments
        test_envs.close()
        return self.metrics


    def dump_plan_video(self, step_before_plan=120): 
        #number of steps before to start to collect frames to dump
        step_before_plan = min(step_before_plan, (self.parms.max_episode_length // self.env.action_repeat))
        
        # Set models to eval mode
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
        video_frames = []
        reward = 0

        with torch.no_grad():
            observation = self.env.reset()
            belief, posterior_state, action = torch.zeros(1, self.parms.belief_size, device=self.parms.device), torch.zeros(1, self.parms.state_size, device=self.parms.device), torch.zeros(1, self.env.action_size, device=self.parms.device)
            tqdm.write("Executing episode.")
            for t in range(step_before_plan): #floor division
                belief, posterior_state, action, next_observation, reward, done, _ = self.update_belief_and_act(self.env,  belief, posterior_state, action, observation.to(device=self.parms.device), [reward], self.env.action_range[0], self.env.action_range[1])
                observation = next_observation
                video_frames.append(make_grid(torch.cat([observation.cpu(), self.observation_model(belief, posterior_state).to(device=self.parms.device).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
                if done:
                    break
            self.create_and_dump_plan(self.env,  belief, posterior_state, action, observation.to(device=self.parms.device), [reward], self.env.action_range[0], self.env.action_range[1])
            
            
        # Set models to train mode
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
        # Close test environments
        self.env.close()

    def create_and_dump_plan(self, env, belief, posterior_state, action, observation, reward, min_action=-inf, max_action=inf): 

        tqdm.write("Dumping plan")
        video_frames = []

        encoded_obs = self.encoder(observation).unsqueeze(dim=0)
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoded_obs)  
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
        next_action,_, beliefs, states, plan = self.planner(belief, posterior_state,False)  # Get action from planner(q(s_t|o≤t,a<t), p)      
        predicted_frames = self.observation_model(beliefs, states).to(device=self.parms.device)

        for i in range(self.parms.planning_horizon):
            plan[i].clamp_(min=env.action_range[0], max=self.env.action_range[1])  # Clip action range
            next_observation, reward, done = env.step(plan[i].cpu())  
            next_observation = next_observation.squeeze(dim=0)
            video_frames.append(make_grid(torch.cat([next_observation, predicted_frames[i]], dim=1) + 0.5, nrow=2).numpy())  # Decentre

        write_video(video_frames, 'dump_plan', self.dump_plan_path, dump_frame=True)  
    
            