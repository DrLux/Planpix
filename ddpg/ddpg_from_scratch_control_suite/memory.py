import random
from collections import namedtuple
import os
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Transition = namedtuple('Transition',
                             ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):

    def __init__(self, capacity, store_path):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.store_path = store_path

        #if folder do not exists, create it
        os.makedirs(self.store_path, exist_ok=True) 

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

