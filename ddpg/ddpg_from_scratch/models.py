import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):

    def __init__(self, state_space, action_space):
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, action_space)
        nn.init.uniform_(self.fc4.weight, -3*1e-3, 3*1e-3) 
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(300)
        self.b3 = nn.BatchNorm1d(400)

    def forward(self, x):

        x = self.b1(F.relu(self.fc1(x)))
        x = self.b2(F.relu(self.fc2(x)))
        x = self.b3(F.relu(self.fc3(x)))
        return torch.tanh(self.fc4(x))

class Critic(nn.Module):

    def __init__(self, state_space, action_space):
        
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fcA1 = nn.Linear(action_space, 256)
        self.fcS1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, 1)
        
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(256)

    def forward(self, state, action):
        
        x = self.b1(F.relu(self.fc1(state)))
        aOut = self.fcA1(F.relu(action))
        sOut = self.b2(F.relu(self.fcS1(x)))
        comb = F.relu(aOut+sOut)
        out = F.relu(self.fc2(comb))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out