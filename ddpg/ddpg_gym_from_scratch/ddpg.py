import gc
import logging
import os
import torch
from models import *

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

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, device, checkpoint_dir=None):

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.device = device
        self.num_actions = self.action_space.shape[0]

        # Define the actor
        self.actor = Actor(num_inputs, self.num_actions).to(device)
        self.actor_target = Actor(num_inputs, self.num_actions).to(device)

        # Define the critic
        self.critic = Critic(num_inputs, self.num_actions).to(device)
        self.critic_target = Critic(num_inputs, self.num_actions).to(device)

        # Define the optimizers for both networks
        self.actor_optimizer  = Adam(self.actor.parameters(),  lr=1e-4,   weight_decay=0.005)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),lr=1e-3, weight_decay=0.005)  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def get_action(self, state, action_noise=None):
        x = state.to(self.device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            mu = action + torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action. Do not use OU noise (https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

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
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()