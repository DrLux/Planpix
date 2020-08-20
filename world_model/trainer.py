from math import inf
from models import * 
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
import os
from torch.distributions.normal import Normal



class Trainer():
    def __init__(self, params, experience_replay_buffer,metrics,results_dir,env):
        self.parms = params     
        self.D = experience_replay_buffer  
        self.metrics = metrics
        self.env = env
        self.tested_episodes = 0    

        self.statistics_path = results_dir+'/statistics'  
        os.makedirs(self.statistics_path, exist_ok=True)
        self.model_path = results_dir+'/model'
        os.makedirs(self.model_path, exist_ok=True) 

        # Create models
        
        # VAE
        self.vae = VAE(self.parms.latent_size).to(device=self.parms.device)
        self.vae_optimiser = optim.Adam(list(self.vae.parameters()), lr=0 if self.parms.learning_rate_schedule != 0 else self.parms.learning_rate, eps=self.parms.adam_epsilon)

        # RNN
        self.mdrnn = MDRNN(self.parms.latent_size, self.env.action_size, self.parms.rnn_hidden_size, self.parms.num_gaussians).to(device=self.parms.device)
        self.mdrnn_optimiser = optim.Adam(list(self.mdrnn.parameters()), lr=0 if self.parms.learning_rate_schedule != 0 else self.parms.learning_rate, eps=self.parms.adam_epsilon)

        # Planner
        self.planner = MPCPlanner(self.env.action_size, self.parms.planning_horizon, self.parms.optimisation_iters, self.parms.candidates, self.parms.top_candidates, self.mdrnn, self.vae, self.parms.device, self.env.action_range[0], self.env.action_range[1])

    def load_checkpoints(self):
        model_path = self.model_path+'/best_model'
        os.makedirs(model_path, exist_ok=True) 
        files = os.listdir(model_path)
        if files:
            checkpoint = [f for f in files if os.path.isfile(os.path.join(model_path, f))]
            model_dicts = torch.load(os.path.join(model_path, checkpoint[0]),map_location=self.parms.device)
            self.vae.load_state_dict(model_dicts['vae'])
            self.vae_optimiser.load_state_dict(model_dicts['vae_optim'])
            print("Loading models checkpoints!")
        else:
            print("Checkpoints not found!")

    def print_debug(self,current_z,current_action,next_z,next_rew, next_flag):
        current_z = current_z.detach()
        current_action = current_action
        mu,sigma,logpi = self.mdrnn.forward(current_z,current_action)
        pred_z, pred_rew, pred_flag = self.mdrnn.get_prediction(mu,sigma,logpi)

        print("next_latent_obs: ", next_z.mean())
        print("pred_latent_obs: ", pred_z.mean())

        print("next_reward: ", next_rew)
        print("pred_reward: ", pred_rew)

        print("next_done: ", next_flag)
        print("pred_flag: ", pred_flag)
    
    def train_models(self):
        tqdm.write("Start training.")
        for episode in tqdm(range(self.parms.num_init_episodes, self.parms.training_episodes) ):
            self.fit_buffer(episode)      
            #self.explore_and_collect(episode)
            #if episode % self.parms.test_interval == 0:
            #torch.save(self.metrics, os.path.join(self.model_path, 'metrics.pth'))
            #torch.save({'vae': self.vae.state_dict(), 'vae_optim': self.vae_optimiser.state_dict()}, os.path.join(self.model_path, 'models_%d.pth' % episode))

    def explore_and_collect(self):
        reward = 0
        # Data collection
        with torch.no_grad():
            done = False
            observation, total_reward = self.env.reset(), 0
            t = 0
            real_rew = []
            predicted_rew = [] 
            total_steps = self.parms.max_episode_length // self.env.action_repeat
            
            for t in range(total_steps):
                action = self.planner.get_action(observation,int(done))
                assert 1 == 2
                next_observation, reward, done = env.step(action)
                self.env.step(action)
                self.D.append(observation, action.cpu(), reward, done)
                total_reward += reward
                observation = next_observation
                if self.parms.flag_render:
                    env.render()
                if done:
                    break

    def fit_buffer(self, episode):
        # Model fitting
        losses = []
        tqdm.write("Training VAE")
        for s in tqdm(range(self.parms.collect_interval)):

            # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
            observations, actions, rewards, nonterminals = self.D.sample(self.parms.batch_size, self.parms.chunk_size)  # Transitions start at time t = 0
            rewards = rewards.unsqueeze(dim=-1)

            # Calculate Loss
            print("observations: ", observations.shape)
            latent_obs = bottle(self.vae.encode, (observations, ))
            '''recon_obs = bottle(self.vae.decode, (latent_obs, ))
            vae_loss = self.vae.get_loss(recon_obs,observations)

            # Update model parameters
            self.vae_optimiser.zero_grad()
            (vae_loss).backward(retain_graph=True) # BACKPROPAGATION
            nn.utils.clip_grad_norm_(list(self.vae.parameters()), self.parms.grad_clip_norm, norm_type=2)
            self.vae_optimiser.step()'''

            ################################################################################

            pred_mus, pred_sigmas, log_pred_pi = self.mdrnn.forward(latent_obs.detach()[0:-1],actions[0:-1])
            mdrnn_loss = self.mdrnn.get_loss(log_pred_pi,pred_mus, pred_sigmas,latent_obs.detach()[1:],rewards[1:],nonterminals[1:])            
            mdrnn_loss = mdrnn_loss / self.parms.latent_size+2

            self.print_debug(latent_obs.detach()[0][0],actions[0][0],latent_obs.detach()[0][1],rewards[0][1], nonterminals[0][1])

            # Update model parameters
            self.mdrnn_optimiser.zero_grad()
            (mdrnn_loss).backward(retain_graph=True) # BACKPROPAGATION
            nn.utils.clip_grad_norm_(list(self.mdrnn.parameters()), self.parms.grad_clip_norm, norm_type=2)
            self.mdrnn_optimiser.step()                     

            print("iterazione: ", s)
            #print("vae_loss: ", vae_loss)
            print("mdrnn_loss: ", mdrnn_loss)
            
            
            ################################
    
            #losses.append([vae_loss.item()])

        #save model and statistics and plot them
        #losses = tuple(zip(*losses))  
        #self.metrics['vae_loss'].append(losses[0])
        #lineplot(self.metrics['episodes'][-len(self.metrics['vae_loss']):], self.metrics['vae_loss'], 'vae_loss', self.statistics_path)

    

