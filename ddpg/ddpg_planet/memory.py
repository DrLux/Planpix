import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_

import torchvision.transforms.functional as TF

import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size) # list of transitions ((o1,a1,r1,..), transitions(o2,a2,r2,..), (ob,ab,rb,...))
        batch = Transition(*zip(*batch)) #concatente list transitions((o1,o2,ob), (a1,a2,ab), ....)
        
        state = torch.cat(batch.state) # single tensor of obs [o1,o2,ob] 
        action = torch.cat(batch.action).reshape(batch_size,-1)        
        done = torch.tensor(batch.done)
        next_state = torch.cat(batch.next_state)
        reward = torch.tensor(batch.reward)

        return [state,action,done,next_state,reward] 

    def __len__(self):
        return len(self.memory)

'''
class ExperienceReplay():
    def __init__(self, size, observation_size, action_size, bit_depth, device,frame_as_state):
        self.device = device
        self.frame_as_state = frame_as_state
        self.size = size
        if self.frame_as_state:
            self.observations = np.empty((size, 3, 64, 64), dtype=np.uint8)
        else:
            self.observations = np.empty((size, observation_size), dtype=np.float32)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32) 
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.bit_depth = bit_depth

    def store_dataset(self,path):
        print("Stroring dataset in: ", path)
        np.savez_compressed(path,
                            observations = self.observations, 
                            actions = self.actions, 
                            rewards = self.rewards, 
                            nonterminal = self.nonterminals,
                            idx = self.idx,
                            full = self.full,
                            steps = self.steps,
                            episodes = self.episodes,
                            bit_depth = self.bit_depth)


    def load_dataset(self,path):
        print("Loadign dataset from: ", path)
        raw_data = np.load(path+"dump_dataset.npz", allow_pickle=True)
        prova = dict(raw_data)
        self.observations = raw_data['observations']
        self.actions = raw_data['actions']
        self.rewards = raw_data['rewards']
        self.nonterminals = raw_data['nonterminal']
        self.idx = raw_data['idx']
        self.full = raw_data['full']
        self.steps = raw_data['steps']
        self.episodes = raw_data['episodes']
        self.bit_depth = raw_data['bit_depth']

    def append(self, observation, action, reward, done):
        if self.frame_as_state: 
            self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
        else:
            self.observations[self.idx] = self.observations[self.idx]
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0 #flag
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    
    
    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, bs, L):
        bs_idxs = []
        for _ in range(bs):
            valid_idx = False
            while not valid_idx:
                idx = np.random.randint(0, self.size if self.full else self.idx - L) #take a random point into the buffer
                idxs = np.arange(idx, idx + L) % self.size # take L sequantial steps from the random point
                valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
            bs_idxs.append(idxs)
        return bs_idxs

    
    ###################
    # Transactions    
    ###################

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if self.frame_as_state:
            preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)   
        
    # Returns a batch of sequence chunks uniformly sampled from the memory
    # n = batch size
    # L = chunk length
    def sample(self, n, L):
        idxs = self._sample_idx(n,L)
        batch = self._retrieve_batch(np.asarray(idxs), n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]
'''