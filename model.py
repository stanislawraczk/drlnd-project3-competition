import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size*2)
        self.fc1 = nn.Linear(state_size*2, fc1_units)
        self.bn2 = nn.BatchNorm1d(fc1_units+action_size*2)
        self.fc2 = nn.Linear(fc1_units+action_size*2, fc2_units)
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = self.bn1(state)
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = self.bn2(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        return self.fc3(x)