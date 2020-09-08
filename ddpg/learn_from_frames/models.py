import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

class Shared_network(nn.Module):
    def __init__(self):
        super(Shared_network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)        
        
        self.fc1 = nn.Linear(23328, 512) #512 real data
        self.b3 = nn.BatchNorm1d(512)

        # Weight Init
        fan_in_uniform_init(self.conv1.weight)
        fan_in_uniform_init(self.conv1.bias)

        fan_in_uniform_init(self.conv2.weight)
        fan_in_uniform_init(self.conv2.bias)

        fan_in_uniform_init(self.conv3.weight)
        fan_in_uniform_init(self.conv3.bias)

        nn.init.uniform_(self.fc1.weight, -3*1e-4, 3*1e-4) 
        nn.init.uniform_(self.fc1.bias, -3*1e-4, 3*1e-4)


    def forward(self, x):
        batch_size = x.size(0)
        x = torch.transpose(x, 1, 3)

        #act = nn.ELU()

        hidden = F.relu(self.conv1(x))
        hidden = F.relu(self.conv2(hidden))
        hidden = F.relu(self.conv3(hidden))
        x = hidden.view(batch_size,-1)

        x = self.b3(F.relu(self.fc1(x)))

        return x


class Actor(nn.Module):

    def __init__(self, state_space, action_space, shared_network):
        
        super(Actor, self).__init__()
        self.shared_network = shared_network
        
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, action_space)
        
        # Batch norm
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(300)
        self.b3 = nn.BatchNorm1d(400)


        # Weight Init
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)

        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)

        fan_in_uniform_init(self.fc3.weight)
        fan_in_uniform_init(self.fc3.bias)

        nn.init.uniform_(self.fc4.weight, -3*1e-4, 3*1e-4) 
        nn.init.uniform_(self.fc4.bias, -3*1e-4, 3*1e-4)

        
    def forward(self, x):
        self.shared_network.eval()
        x = self.shared_network.forward(x)
        x = self.b1(F.relu(self.fc1(x)))
        x = self.b2(F.relu(self.fc2(x)))
        x = self.b3(F.relu(self.fc3(x)))
        result = torch.tanh(self.fc4(x)) 
        self.shared_network.train()
        return result
        

        

class Critic(nn.Module):

    def __init__(self, state_space, action_space, shared_network):
        
        super(Critic, self).__init__()
        self.shared_network = shared_network
        
        self.fc1 = nn.Linear(512, 128)
        self.fcA1 = nn.Linear(action_space, 256)
        self.fcS1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, 1)
        
        # Batch norm
        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(256)

        # Weight Init
        fan_in_uniform_init(self.fc1.weight)
        fan_in_uniform_init(self.fc1.bias)

        fan_in_uniform_init(self.fcA1.weight)
        fan_in_uniform_init(self.fcA1.bias)

        fan_in_uniform_init(self.fcS1.weight)
        fan_in_uniform_init(self.fcS1.bias)

        fan_in_uniform_init(self.fc2.weight)
        fan_in_uniform_init(self.fc2.bias)

        fan_in_uniform_init(self.fc3.weight)
        fan_in_uniform_init(self.fc3.bias)

        nn.init.uniform_(self.fc4.weight, -3*1e-4, 3*1e-4) 
        nn.init.uniform_(self.fc4.bias, -3*1e-4, 3*1e-4)

        
    def forward(self, state, action):
        x = self.shared_network.forward(state)
        x = self.b1(F.relu(self.fc1(x)))
        aOut = self.fcA1(F.relu(action))
        sOut = self.b2(F.relu(self.fcS1(x)))
        comb = F.relu(aOut+sOut)
        out = F.relu(self.fc2(comb))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

        