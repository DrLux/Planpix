import random
from collections import namedtuple
import numpy as np
import torch


class ReplayMemory(object):

    def __init__(self, capacity, action_size):
        self.observations = np.empty((capacity, 9, 64, 64), np.uint8)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, ), dtype=np.float32) 
        self.nonterminals = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.idx = 0
        self.full = False
        
    # Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    def postprocess_observation(self,observation):
        return np.clip(np.floor((observation + 0.5) * 2 ** self.bit_depth) * 2 ** (8 - self.bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)

    def push(self, state, action, done,reward):        
        #self.observations[self.idx] = self.postprocess_observation(state)  # Decentre and discretise visual observations (to save memory)
        self.observations[self.idx] = state  # Decentre and discretise visual observations (to save memory)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_batch(self, batch_size):
        rand_idxs = np.random.randint(0, self.capacity if self.full else self.idx,batch_size)
        return [self.observations[rand_idxs],self.actions[rand_idxs],self.rewards[rand_idxs],self.nonterminals[rand_idxs]]

    def __len__(self):
        return self.capacity if self.full else self.idx