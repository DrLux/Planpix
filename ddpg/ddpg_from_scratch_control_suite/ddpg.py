import gc
import logging
import os
import torch
from models import *
from noise import OrnsteinUhlenbeckActionNoise
import os


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

    def __init__(self,env,parms):

        self.gamma = parms.gamma
        self.tau = parms.tau
        self.min_action,self.max_action = env.action_range()
        self.device = parms.device
        self.num_actions = env.action_space()
        self.size_state = env.state_space()
        self.noise_stddev = parms.noise_stddev
        self.checkpoint_path = parms.checkpoint_dir
 
         

        # Define the actor
        self.actor = Actor(self.size_state, self.num_actions).to(self.device)
        self.actor_target = Actor(self.size_state, self.num_actions).to(self.device)

        # Define the critic
        self.critic = Critic(self.size_state, self.num_actions).to(self.device)
        self.critic_target = Critic(self.size_state, self.num_actions).to(self.device)

        # Define the optimizers for both networks
        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=parms.actor_lr )                          # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(), lr=parms.critic_lr,   weight_decay=parms.weight_decay)  # optimizer for the critic network

        self.hard_swap()

        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.num_actions),
                                            sigma=float(self.noise_stddev) * np.ones(self.num_actions))
        self.ou_noise.reset()

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
        else:
            noise = 0
            
        mu += noise  

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.min_action,self.max_action)

        return mu

    def update_params(self, batch):
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        for param in self.actor.parameters():
                param.grad.data.clamp_(-1, 1)
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