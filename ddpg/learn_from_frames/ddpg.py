import gc
import logging
import os
import torch
from models import *
from noise import OrnsteinUhlenbeckActionNoise

from torchvision import transforms
import torch


import torch.nn.functional as F
from torch.optim import Adam

from models import Actor,Critic

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class DDPG(object):

    def __init__(self, gamma, tau,num_inputs, env,device, checkpoint_dir=None):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.min_action,self.max_action = self.env.action_range()
        self.device = device
        self.num_actions = self.env.action_space()
        self.noise_stddev = 0.3

        self.results_path = '/home/luca/Desktop/luca/ddpg/'
        self.checkpoint_path = os.path.join(self.results_path, 'checkpoint/')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Define Shared Network
        self.shared_network = Shared_network().to(device)

        # Define the actor
        self.actor = Actor(num_inputs, self.num_actions,self.shared_network).to(device)
        self.actor_target = Actor(num_inputs, self.num_actions,self.shared_network).to(device)

        # Define the critic
        self.critic = Critic(num_inputs, self.num_actions,self.shared_network).to(device)
        self.critic_target = Critic(num_inputs, self.num_actions,self.shared_network).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=1e-4 )                          # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(), lr=1e-4,   weight_decay=0.002)  # optimizer for the critic network

        self.hard_swap()

        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.num_actions),
                                            sigma=float(self.noise_stddev) * np.ones(self.num_actions))
        self.ou_noise.reset()

    def eval_mode(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic.eval()
        self.shared_network.eval()

    def train_mode(self):
        self.actor.train()
        self.actor_target.train()
        self.critic_target.train()
        self.critic.train()
        self.shared_network.train()


    def get_action(self, state, episode, action_noise=True):
        x = state.to(self.device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise:
            noise = torch.Tensor(self.ou_noise.noise()).to(self.device) * 1.0/(1.0 + 0.005*episode)
            noise = noise.clamp(0,0.2)
            mu = mu + noise  # Add exploration noise ε ~ p(ε) to the action. Do not use OU noise (https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.min_action,self.max_action)

        return mu

    def update_params(self, batch):

        # Get tensors from the batch
        
        # next_state_batch = batch_state[1:]
        # state_batch = batch_state[0:-1]

        batch_state = torch.Tensor(batch[0]).to(self.device) 
        batch_act = torch.Tensor(batch[1]).to(self.device)
        batch_rwd = torch.Tensor(batch[2]).to(self.device)
        batch_done = torch.Tensor(batch[3]).to(self.device)

        #self.env.quantise_frame(state_batch.float())
        #self.env.quantise_frame(next_state_batch.float())
        
        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(batch_state[1:]) #next_state
        next_state_action_values = self.critic_target(batch_state[1:], next_action_batch.detach()) #next_state

        # Compute the target
        batch_rwd = batch_rwd.unsqueeze(1)
        #batch_done = batch_done.unsqueeze(1)
        expected_values = batch_rwd[:-1] + (1.0 - batch_done[:-1]) * self.gamma * next_state_action_values
        
        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(batch_state[:-1], batch_act[:-1])
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(batch_state[:-1], self.actor(batch_state[:-1]))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp(-1, 1)
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()
    
    def hard_swap(self):
        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def store_model(self):
        print("Storing model at: ", self.checkpoint_path)
        checkpoint = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'criti_optim': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, 'checkpoint.pth') )

    def load_model(self):
        files = os.listdir(self.checkpoint_path)
        if files:
            print("Loading models checkpoints!")
            model_dicts = torch.load(os.path.join(self.checkpoint_path, 'checkpoint.pth'),map_location=self.device)
            self.actor.load_state_dict(model_dicts['actor'])
            self.actor_optimizer.load_state_dict(model_dicts['actor_optim'])
            self.critic.load_state_dict(model_dicts['critic'])
            self.critic_optimizer.load_state_dict(model_dicts['criti_optim'])
        else:
            print("Checkpoints not found!")