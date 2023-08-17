"""
Replay Buffer
Author: Sameera Lanka
Website: https://sameera-lanka.com
Modified for DIRL
"""

import random
import torch
from collections import deque


class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)

    def __len__(self):
        return len(self.data)

    def sample_batch(self, batchSize):
        if len(self.data) < batchSize:
            print('Not enough entries to sample without replacement.')
            return None
        else:
            batch = random.sample(self.data, batchSize)
            curState = torch.cat([element[0].view(1, -1) for element in batch])
            action = torch.cat([element[1].view(1, -1) for element in batch])
            nextState = torch.cat([element[2].view(1, -1) for element in batch])
            reward = [element[3] for element in batch]
            terminal = [element[4] for element in batch]
        return curState, action, nextState, reward, terminal

    def append(self, element):
        self.data.append(element)
