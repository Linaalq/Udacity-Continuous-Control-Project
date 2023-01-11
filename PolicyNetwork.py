import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):

    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        state_size = config['environment']['state_size']
        action_size = config['environment']['action_size']
        hidden_size = config['hyperparameters']['hidden_size']
        device = config['pytorch']['device']
        
        # actor network
        self.fc1_actor = nn.Linear(state_size, hidden_size)
        self.fc2_actor = nn.Linear(hidden_size, hidden_size)
        self.fc3_actor = nn.Linear(hidden_size, action_size)
        
        #critic network
        self.fc1_critic = nn.Linear(state_size, hidden_size)
        self.fc2_critic = nn.Linear(hidden_size, hidden_size)
        self.fc3_critic = nn.Linear(hidden_size, 1)
        
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.to(device)

    def forward(self, x, action=None):
        x = torch.Tensor(x)
        
        actor = F.relu(self.fc1_actor(x))
        actor = F.relu(self.fc2_actor(actor))
        mean = torch.tanh(self.fc3_actor(actor))

        critic = F.relu(self.fc1_critic(x))
        critic = F.relu(self.fc2_critic(critic))
        v = self.fc3_critic(critic)
        
        dist = torch.distributions.Normal(mean, self.std)
        
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = torch.Tensor(np.zeros((log_prob.size(0), 1)))
        return action, log_prob, entropy, v