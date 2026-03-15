"""
Replay Buffer for Experience Replay.
Stores past transitions to break temporal correlation during training.
"""
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size=10000):
        """
        Initialize a circular buffer using a deque.
        """
        self.buffer = deque(maxlen=size)

    def push(self, transition):
        """
        Add a new experience (s, a, r, s') to the buffer.
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences for stochastic gradient descent.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
