from abc import ABC, abstractmethod
import numpy as np

class State():
    def __init__(self, observation, reward=0, done=False, info=""):
        self.observation = observation
        self.neighborhood = {}
        self.reward = reward
        self.done = done
        self.info = info

    def step(self, action):
        if action in self.neighborhood:
            return self.neighborhood[action]
        else:
            return self

class World(ABC):
    @abstractmethod
    def __init__(self):
        self.states = []
        self.state = None

    def step(self, action):
        self.state = self.state.step(action)
        return self.state.observation, self.state.reward, self.state.done, self.state.info

    @abstractmethod
    def reset(self):
        self.__init__()


class OneHot1DWorld(World):
    def __init__(self, size):
        super().__init__()

        # create state space with observations
        for i in range(size):
            obs = np.zeros(size)
            obs[i] = 1
            self.states.append(State(obs))

        # add transitions between them via action 0 (next) and 1 (previous)
        for i in range(size):
            # action 0
            if i < size - 1:
                self.states[i].neighborhood[0] = self.states[i + 1]

            # action 1
            if i > 0:
                self.states[i].neighborhood[1] = self.states[i - 1]

        # init at first state
        self.state = self.states[0]

    def reset(self):
        self.__init__(len(self.states))

class OneHot1DCyclicWorld(World):
    def __init__(self, size):
        super().__init__()

        # TODO implement the init here

class OneHot2DWorld(World):
    # TODO design intuitive way how to provide world_definition
    def __init__(self, world_definiton):
        super().__init__()

        # TODO implement the init here


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
    @abstractmethod
    def train(self):
        pass

    def decide(self, observation):
        return np.random.randint(self.n_actions)
