# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_size, action_size, n_agents, seed):

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.bn0 = nn.BatchNorm1d((state_size+action_size)*n_agents)
        self.fcs1 = nn.Linear( (state_size+action_size)*n_agents, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)


    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        state = self.bn0(x)
        x = F.selu(self.fcs1(state))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        return  F.selu(self.fc5(x))