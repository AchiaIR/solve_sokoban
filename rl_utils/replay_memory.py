"""Define the replay memory buffer which stores past experiences
to reduce the correlation between successive updates and to improve stability."""

import random
from collections import namedtuple, deque


class ReplayMemory(object):
    "stores past experiences as fifo"

    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.memory_baffer = deque([], maxlen=capacity)
        self.transition = namedtuple('T', ('s', 'a', 'ns', 'r'))

    def __len__(self):
        return len(self.memory_baffer)

    def push(self, state, action, next_state, reward):
        transition = (state, action, next_state, reward)
        self.memory_baffer.append(transition)

    def sample(self):
        batch = random.sample(self.memory_baffer, self.batch_size)
        return self.transition(*zip(*batch))
