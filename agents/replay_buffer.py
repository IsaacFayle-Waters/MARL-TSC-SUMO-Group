import random
from collections import deque

class ReplayBuffer:

    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
