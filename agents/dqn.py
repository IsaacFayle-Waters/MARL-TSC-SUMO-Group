"""
Deep Q-Network (DQN) Implementation.
This model approximates the Q-value function for traffic signal control decisions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize the Neural Network.
        Args:
            state_size (int): Number of input features (Density/Queuing metrics).
            action_size (int): Number of possible traffic light phases.
        """
        super(DQN, self).__init__()
        # Standard multi-layer perceptron architecture
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        Forward pass to predict Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
