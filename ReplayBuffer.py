from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size              #buffer_size=10000 
        self.num_experiences = 0
        self.buffer = deque()
        self.mean_reward = 0.0    #1、添加了这个

    def getBatch(self, batch_size):                #when the agent is trainning, and need samples.
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        self.mean_reward = ( (self.mean_reward * (self.num_experiences-1)) + reward) / float( self.num_experiences ) #2、every step's average reward: sum of reward / sum of steps.

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def getMeanReward(self):                                                #3、acquire mean_reward.
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.mean_reward

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
