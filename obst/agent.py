from abc import ABC, abstractmethod
import numpy as np

class AbstractAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def behave(self, observation):
        pass

    @abstractmethod
    def reset(self):
        pass


class BufferedAgent(AbstractAgent):
    def __init__(self, buffer_size, training_period, n_actions):
        self.buffer_size = buffer_size
        self.buffer = []

        self.training_period = training_period
        self.step = 0

        self.n_actions = n_actions

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def decide(self, observation):
        pass

    @abstractmethod
    def reset(self):
        self.buffer = []
        self.step = 0

    def observe(self, observation):
        self.step += 1

        self.buffer.append(observation)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if self.step % self.training_period == 0:
            self.train()

    def behave(self, observation):
        self.observe(observation)
        return self.decide(observation)


class RandomBufferedAgent(BufferedAgent):
    # @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    def reset(self):
        super().reset()

    def decide(self, observation):
        return np.random.randint(self.n_actions)
