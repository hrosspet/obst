from abc import ABC, abstractmethod
import numpy as np
import random
import math
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

REWARD_IDX = 763

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
        self.states[REWARD_IDX].reward = 1

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

class My2DWorld(World):
    def __init__(self, width, height):
        super().__init__()

        self.width = width
        self.height = height
        self.cyclic = True

        self.reset(False)

    def step(self, action):
        # Respond to action
        if action == 0:
            if self.agt_y > 0:
                self.agt_y -= 1
            elif self.cyclic:
                self.agt_y = self.height - 1
        if action == 1:
            if self.agt_x < self.width:
                self.agt_x += 1
            elif self.cyclic:
                self.agt_x = 0
        if action == 2:
            if self.agt_y < self.height:
                self.agt_y += 1
            elif self.cyclic:
                self.agt_y = 0
        if action == 3:
            if self.agt_x > 0:
                self.agt_x -= 1
            elif self.cyclic:
                self.agt_y = self.width - 1
        print(self.agt_x, self.agt_y)
        # An arbitary set of numbers that changes for each state
        obs = (math.log(self.agt_x+1), math.log(self.agt_x+1, self.agt_y+2), math.log(self.width - self.agt_x+1, 10), math.log(self.height - self.agt_y+1), math.log(abs(self.agt_y - self.agt_x)+1))
        return obs, 0, False, None

    def reset(self, test):
        self.agt_x = self.width  // 2
        self.agt_y = self.height // 2

from datetime import datetime
import matplotlib.pyplot as plt

class Visualizing2DWorld(My2DWorld):
    def __init__(self, width, height):
        super().__init__(width, height)

        self.step_no = 0
        self.pos_history = []   # We need to keep position history so that we can draw the agent's movements. Although the agent has is's own buffer, it doesn't contain the x and y -- only the observation from the given point.

    def step(self, action):
        # Save current coordinates
        self.pos_history.append((self.agt_x, self.agt_y))

        return super().step(action)

    def plot_actions(self):
        vis = plt.subplot()    # Visualization

        vis.axis((0, self.width, 0, self.height))
        vis.grid(True)

        for i in range(len(self.pos_history) - 1):
            pct = i / len(self.pos_history);

            x_old, y_old = self.pos_history[i - 1]
            x_new, y_new = self.pos_history[i]

            vis.plot([x_old, x_new], [y_old, y_new], color=(pct, 0, 1-pct))

        self.pos_history.clear()
