# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):
    def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.reset_parameters()

    
    def forward(self, states, actions):
        x_state_action = torch.cat((states, actions), dim=1)
        
        x = F.selu(self.fc1(x_state_action))
        x = self.bn1(x)
        x = F.selu(self.fc2(x))
        x = self.bn2(x)
        x = F.selu(self.fc3(x))
        x = self.bn3(x)
        x = F.selu(self.fc4(x))
        x = self.bn4(x)
        x = torch.tanh(self.fc5(x))
        
        return x
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
