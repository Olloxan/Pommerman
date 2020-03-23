import numpy as np
import torch
import torch.nn as nn
from common.OnPolicy import OnPolicy


class ActorCritic(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()
        
        self.in_shape = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = 16

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        fc_size = 256
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), fc_size),
            nn.ReLU(),
        )
        
        self.critic  = nn.Linear(fc_size, 1)
        self.actor   = nn.Linear(fc_size, num_actions)
        
    def forward(self, x):            
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def feature_size(self):       
        convoutput1 = self.calculate_conv_output(self.in_shape[1:3], self.out_channels, 3)       
        convoutput2 = self.calculate_conv_output(convoutput1[1:3], self.out_channels, 3, 2)
        features = int(np.prod(convoutput2))
        return features

  
class ActorCritic_Large(OnPolicy):
    def __init__(self, in_shape, num_actions):
        super(ActorCritic_Large, self).__init__()

        self.in_shape = in_shape
        self.in_channels = in_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        fc_size = 512        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), fc_size),
            nn.ReLU()
        )

        self.critic  = nn.Linear(fc_size, 1)
        self.actor   = nn.Linear(fc_size, num_actions)

    def forward(self, x):            
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value

    def feature_size(self):       
        convoutput1 = self.calculate_conv_output(self.in_shape[1:3], 32, 8, 4)       
        convoutput2 = self.calculate_conv_output(convoutput1[1:3], 64, 4, 2)
        convoutput3 = self.calculate_conv_output(convoutput2[1:3], 64, 3, 1)
        features = int(np.prod(convoutput3))
        return features

class RolloutStorage(object):
    def __init__(self, num_steps, num_envs, state_shape):
        self.num_steps = num_steps
        self.num_envs  = num_envs
        self.states  = torch.zeros(num_steps + 1, num_envs, *state_shape)
        self.rewards = torch.zeros(num_steps,     num_envs, 1)
        self.masks   = torch.ones(num_steps  + 1, num_envs, 1)
        self.actions = torch.zeros(num_steps,     num_envs, 1).long()
        self.use_cuda = False
            
    def cuda(self):
        self.use_cuda  = True
        self.states    = self.states.cuda()
        self.rewards   = self.rewards.cuda()
        self.masks     = self.masks.cuda()
        self.actions   = self.actions.cuda()
        
    def insert(self, step, state, action, reward, mask):
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        
    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        
    def compute_returns(self, next_value, gamma):
        returns   = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        if self.use_cuda:
            returns = returns.cuda()
        returns[-1] = next_value
        for step in reversed(range(self.num_steps)):
            returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return returns[:-1]
