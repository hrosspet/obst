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

# rewards = [(20, 19), (34, 28), (21, 33)]
rewards = [(2, 2)]

class My2DWorld(World):
    def __init__(self, width, height, cyclic=False):
        super().__init__()

        self.width = width
        self.height = height
        self.cyclic = cyclic

        self.reset(False)

    def step(self, action):
        # Respond to action
        if action == 0:
            if self.agt_y > 0:
                self.agt_y -= 1
            elif self.cyclic:
                self.agt_y = self.height - 1
        if action == 1:
            if self.agt_x < self.width - 1:
                self.agt_x += 1
            elif self.cyclic:
                self.agt_x = 0
        if action == 2:
            if self.agt_y < self.height - 1:
                self.agt_y += 1
            elif self.cyclic:
                self.agt_y = 0
        if action == 3:
            if self.agt_x > 0:
                self.agt_x -= 1
            elif self.cyclic:
                self.agt_y = self.width - 1

        logger.debug("{} {}".format(self.agt_x, self.agt_y))

        # An arbitary set of numbers that changes for each state
        obs = np.array([math.log(self.agt_x+1), math.log(self.agt_x+1, self.agt_y+2), math.log(self.width - self.agt_x+1, 10), math.log(self.height - self.agt_y+1), math.log(abs(self.agt_y - self.agt_x)+1)])
        # obs=np.array([self.agt_x, self.agt_y])
        reward: bool = (self.agt_x, self.agt_y) in rewards
        return obs, (1 if reward else 0), reward, None  # reset on reward

    def reset(self, test=False):
        self.agt_x = self.width  // 2
        self.agt_y = self.height // 2

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class Visualizing2DWorld(My2DWorld):
    def __init__(self, width, height, cyclic):
        super().__init__(width, height, cyclic)

        self.step_no = 0
        self.pos_history = []   # We need to keep position history so that we can draw the agent's movements. Although the agent has is's own buffer, it doesn't contain the x and y -- only the observation from the given point.
        self.heatmap = np.zeros((height, width))
        self.last_all_visited = self.heatmap.copy()
        self.all_visited_log = []
        self.reset_log = []  # Log every time the world resets

    def step(self, action):
        self.step_no += 1

        # Save current coordinates
        self.pos_history.append((self.agt_x, self.agt_y))

        # Update heat map
        self.heatmap[self.agt_y, self.agt_x] += 1

        if np.min(self.heatmap - self.last_all_visited) != 0:   # If all tiles have been visited since last record
            self.all_visited_log.append(self.step_no)
            self.last_all_visited = self.heatmap.copy()
            for coords in rewards:
                self.last_all_visited[coords] += 1     # THe coords of the reward are never visited

        ret = super().step(action)

        if ret[2] == True:
            self.reset_log.append(self.step_no)

        return ret

    def plot_actions(self):
        vis = plt.subplot(1, 4, 1)    # Visualization

        vis.axis((0, self.width - 1, 0, self.height - 1))
        vis.grid(True)

        # Draw rewards
        for coords in rewards:
            vis.scatter(*coords, c='green')

        # Draw steps
        for i in range(len(self.pos_history) - 1):
            pct = i / len(self.pos_history);

            x_old, y_old = self.pos_history[i - 1]
            x_new, y_new = self.pos_history[i]

            vis.plot([x_old, x_new], [y_old, y_new], color=(pct, 0, 1-pct))

        self.pos_history.clear()

    def plot_resets(self):
        diffs = [s2 - s1 for s1, s2 in zip(self.reset_log[:-1], self.reset_log[1:])]

        grph = plt.subplot(1, 4, 2)
        plt.xlim(0, len(diffs))
        grph.plot(range(len(diffs)), diffs, c='blue')
        plt.title('steps between restarts')

    def plot_all_visited(self):
        diffs = [s2 - s1 for s1, s2 in zip(self.all_visited_log[:-1], self.all_visited_log[1:])]

        grph = plt.subplot(1, 4, 3)
        plt.xlim(0, len(diffs))
        grph.plot(range(len(diffs)), diffs, c='blue')
        plt.title('steps between all fields being visited')

    def plot_heatmap(self):
        plt.subplot(1, 4, 4)

        # htmp = self.heatmap.copy()
        # htmp /= np.max(htmp)        # make all values between 0 and 1

        hmap = plt.imshow(self.heatmap, cmap='Greys', origin='lower', norm=Normalize(vmin=0, vmax=np.max(self.heatmap)))
        plt.title('heatmap')
        plt.colorbar(hmap)

    def plot(self):
        self.plot_actions()
        self.plot_resets()
        self.plot_all_visited()
        self.plot_heatmap()
