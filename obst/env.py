from abc import ABC, abstractmethod
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

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
        logger.debug('size: %d', size)

        # create state space with observations
        for i in range(size):
            obs = np.zeros(size)
            obs[i] = 1
            self.states.append(State(obs))  # Create a new state from the given observatrion and append it to the list

        # add transitions between them via action 0 (next) and 1 (previous)
        for i in range(size):
            # action 0
            if i < size - 1:
                self.states[i].neighborhood[0] = self.states[i + 1]

            # action 1
            if i > 0:
                self.states[i].neighborhood[1] = self.states[i - 1]

        # Set reward for final (exit) state
        for f in range(size):
            self.states[f].reward = f % 2

        # init at first state
        self.state = self.states[0]

    def reset(self, test=False):
        if test:
            self.state = self.states[-1]
        else:
            self.state = self.states[0]
            # self.state = random.choice(self.states)

class OneHot1DCyclicWorld(OneHot1DWorld):
    def __init__(self, size):
        super().__init__(size)

        self.states[0].neighborhood[1] = self.states[-1]
        self.states[-1].neighborhood[0] = self.states[0]

class OneHot2DWorld(World):
    # TODO design intuitive way how to provide world_definition
    def __init__(self, world_definiton):
        super().__init__()

        # TODO implement the init here
