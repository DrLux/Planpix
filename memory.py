import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_

import torchvision.transforms.functional as TF

class ExperienceReplay():
    def __init__(self, size, observation_size, action_size, bit_depth, device):
        self.device = device
        self.size = size
        self.observations = np.empty((size, 3, 64, 64), dtype=np.uint8)
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size, ), dtype=np.float32) 
        self.nonterminals = np.empty((size, 1), dtype=np.float32)
        self.tbs = 0 #trajectory batch size
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.bit_depth = bit_depth

    def append(self, observation, action, reward, done):
        self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)  # Decentre and discretise visual observations (to save memory)
        self.actions[self.idx] = action.numpy()
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0 #flag
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

    
    
    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def _sample_idx(self, bs, L):
        bs_idxs = []
        bs_next_idxs = []
        for _ in range(bs):
            valid_idx = False
            while not valid_idx:
                idx = np.random.randint(0, self.size if self.full else self.idx - L) #take a random point into the buffer
                idxs = np.arange(idx, idx + L) % self.size # take L sequantial steps from the random point
                next_idxs = np.arange(idx+1, idx + L+1) % self.size # get the next steps
                valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
                valid_idx = not self.idx in next_idxs[1:]  
            bs_idxs.append(idxs)
            bs_next_idxs.append(next_idxs)
        return bs_idxs,bs_next_idxs

    ###################
    # Trajectories    
    ###################

    # input index bs,chunk_size
    # output data chunk_size,bs
    def _retrieve_trajectories(self, idxs, next_idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        vec_next_idxs = next_idxs.transpose().reshape(-1)  # Unroll indices

        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations

        next_observation = torch.as_tensor(self.observations[vec_next_idxs].astype(np.float32))
        preprocess_observation_(next_observation, self.bit_depth)  # Undo discretisation for visual observations

        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), next_observation.reshape(L, n, *next_observation.shape[1:]), self.nonterminals[vec_idxs].reshape(L, n, 1), self.nonterminals[vec_next_idxs].reshape(L, n, 1)

    # Returns a batch of sequence chunks uniformly sampled from the memory
    # n = batch size
    # L = chunk length
    def get_trajectories(self, n, L):
        idxs, next_idxs = self._sample_idx(n,L)
        batch = self._retrieve_trajectories(np.asarray(idxs), np.asarray(next_idxs), n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch] 

    ###################
    # Transactions    
    ###################

    def _retrieve_batch(self, idxs, n, L):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.nonterminals[vec_idxs].reshape(L, n, 1)   
        
    # Returns a batch of sequence chunks uniformly sampled from the memory
    # n = batch size
    # L = chunk length
    def sample(self, n, L):
        idxs,_ = self._sample_idx(n,L)
        batch = self._retrieve_batch(np.asarray(idxs), n, L)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]