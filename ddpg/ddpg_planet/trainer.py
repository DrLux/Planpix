from math import inf
import numpy as np
from torch import nn, optim
from planner import MPCPlanner
import random
from torch.distributions import Normal
from torch.nn import functional as F
from torch.distributions.kl import kl_divergence
from utils import lineplot, write_video, double_lineplot
from torchvision.utils import make_grid, save_image
from env import EnvBatcher,ControlSuiteEnv
import os
from tqdm import tqdm
import os
import torch
from agent import *
import torch
from models import Actor,Critic
from torch.optim import Adam
import torch.nn.functional as F

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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
        
        #if folder do not exists, create it
        os.makedirs(self.statistics_path, exist_ok=True) 
        os.makedirs(self.model_path, exist_ok=True) 
        os.makedirs(self.video_path, exist_ok=True) 
        os.makedirs(self.rew_vs_pred_rew_path, exist_ok=True)

        #Create agent
        # Create models
        # Define the actor
        self.actor = Actor(self.env.observation_size, self.env.action_size).to(device=self.parms.device)
        self.actor_target = Actor(self.env.observation_size, self.env.action_size).to(device=self.parms.device)

        # Define the critic
        self.critic = Critic(self.env.observation_size, self.env.action_size).to(device=self.parms.device)
        self.critic_target = Critic(self.env.observation_size, self.env.action_size).to(device=self.parms.device)

        # Define the optimizers for both networks
        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=self.parms.actor_lr,   weight_decay=self.parms.weight_decay)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.parms.critic_lr, weight_decay=self.parms.weight_decay)  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def load_checkpoints(self):
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

    def get_action(self, state, action_noise=False):
        # Get the continous action value to perform in the env
        self.actor.eval() 
        action = self.actor(x)
        self.actor.train() 
        action = action.data

        # During training we add noise for exploration
        if action_noise:
            action = action + torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action. Do not use OU noise (https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

        # Clip the output according to the action space of the env
        action = action.clamp(self.env.action_range())
        return action
    
    def fit_buffer(self,episode):
        # Model fitting
        losses = []
        tqdm.write("Fitting buffer")
        for s in tqdm(range(self.parms.collect_interval)):
            # Sample from buffer
            state,action,terminal,next_state,reward = self.D.sample(self.parms.batch_size)  
            
            # Get the actions and the state values to compute the targets
            next_action = self.actor_target(next_state)
            next_state_action_values = self.critic_target(next_state, next_action.detach())

            # Compute the target
            reward = reward.unsqueeze(1) 
            done = done.unsqueeze(1) 
            expected_values = reward + (1.0 - done) * self.parms.discount_factor * next_state_action_values

            # Update the critic network
            self.critic_optimizer.zero_grad()
            critic_loss = self.critic.get_loss(state,action,expected_values.detach())
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss = -self.critic(state, self.actor(state))
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            soft_update(self.actor_target, self.actor, self.parms.update_factor)
            soft_update(self.critic_target, self.critic, self.parms.update_factor)

            # Update actor and critic according to the batch
            losses.append([actor_loss.item(), critic_loss.item()])#, regularizer_loss.item()])

        #save statistics and plot them
        losses = tuple(zip(*losses))  
        self.metrics['actor_loss'].append(losses[0])
        self.metrics['critic_loss'].append(losses[1])
      
        lineplot(self.metrics['episodes'][-len(self.metrics['actor_loss']):], self.metrics['actor_loss'], 'actor_loss', self.statistics_path)
        lineplot(self.metrics['episodes'][-len(self.metrics['critic_loss']):], self.metrics['critic_loss'], 'critic_loss', self.statistics_path)
        
    def explore_and_collect(self,episode):
        tqdm.write("Collect new data:")
        # Data collection
        with torch.no_grad():
            done = False
            observation, total_reward = self.env.reset(), 0
            total_steps = self.parms.max_episode_length // self.env.action_repeat

            for t in tqdm(range(total_steps)):
                # QUI INVECE ESPLORI
                action = self.get_action(observation,action_noise=True)
                next_observation, reward, done = self.env.step(action)
                self.D.append(observation,action.cpu(),done,next_observation,reward)
                total_reward += reward
                observation = next_observation
                if done:
                    break

        # Update and plot train reward metrics
        self.metrics['steps'].append( (t * self.env.action_repeat) + self.metrics['steps'][-1])
        self.metrics['episodes'].append(episode)
        self.metrics['train_rewards'].append(total_reward)
        lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.statistics_path)

    def train_models(self):
        # da (num_episodi_per_inizializzare) a (training_episodes + episodi_per_inizializzare)
        tqdm.write("Start training.")
        for episode in tqdm(range(self.parms.num_init_episodes +1, self.parms.training_episodes) ):
            self.fit_buffer(episode)       
            self.explore_and_collect(episode)
            #if episode % self.parms.test_interval == 0:
            #    self.test_model(episode)
            #    torch.save(self.metrics, os.path.join(self.model_path, 'metrics.pth'))
            #    torch.save({'transition_model': self.transition_model.state_dict(), 'observation_model': self.observation_model.state_dict(), 'reward_model': self.reward_model.state_dict(), 'encoder': self.encoder.state_dict(), 'optimiser': self.optimiser.state_dict()},  os.path.join(self.model_path, 'models_%d.pth' % episode))
            
            #creare la variabile self parms dataset_interval
            #if episode % self.parms.storing_dataset_interval == 0:
            #    self.D.store_dataset(self.parms.dataset_path+'dump_dataset')

        return self.metrics

    '''def test_model(self, episode=None): #no explore here
        if episode is None:
            episode = self.tested_episodes


        # Set models to eval mode
        self.transition_model.eval()
        self.observation_model.eval()
        self.reward_model.eval()
        self.encoder.eval()
        #self.regularizer.eval()
        
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
            for t in range(total_steps): #floor division    
                #print("iterazione: ", t)
                belief, posterior_state, action, next_observation, rewards, done, pred_next_rew  = self.update_belief_and_act(test_envs,  belief, posterior_state, action, observation.to(device=self.parms.device), list(rewards), self.env.action_range[0], self.env.action_range[1])
                total_rewards += rewards.numpy()
                real_rew[t] = rewards
                predicted_rew[t]  = pred_next_rew
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
        
        for i in range(self.parms.test_episodes):
            double_lineplot(np.arange(total_steps), real_rew[i], predicted_rew[i], "Real vs Predicted rewards_ep%s_test_%s" %(str(episode),str(i+1)), self.statistics_path)

        write_video(video_frames, 'test_episode_%s' % str(episode), self.video_path)  # Lossy compression
        # Set models to train mode
        self.transition_model.train()
        self.observation_model.train()
        self.reward_model.train()
        self.encoder.train()
        # Close test environments
        test_envs.close()
        return self.metrics
    '''


    
    