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

    def close(self):
        pass

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
        # obs = np.array([
        #     math.log(self.agt_x+1),
        #     math.log(self.agt_x+1, self.agt_y+2),
        #     math.log(self.width - self.agt_x+1, 10),
        #     math.log(self.height - self.agt_y+1),
        #     math.log(abs(self.agt_y - self.agt_x)+1)
        # ])
        obs = np.array([
            3 * self.agt_x + 3,
            7 * self.agt_x + 7 + 5 * self.agt_y + 2,
            5 * self.width - 3 * self.agt_x + 10,
            13 * self.height - 7 * self.agt_y + 7,
            3 * self.agt_y - 17 * self.agt_x - 5
        ])
        # obs=np.array([self.agt_x, self.agt_y])
        reward = (self.agt_x, self.agt_y) in rewards
        return obs, (1 if reward else 0), reward, None  # reset on reward

    def reset(self, test=False):
        self.agt_x = self.width  // 2 - 1
        self.agt_y = self.height // 2 - 1


class Twisted2DWorld(World):
    def __init__(self, world_file):
        super().__init__()

        # read world definition from given text file
        self._read_def(world_file)
        self.width = len(self.world_map[0])
        self.height = len(self.world_map)

        self.reset(False)

        # self._calculate_observations()    # Real slow on big maps

    def _read_def(self, world_file):
        with open(world_file, 'r') as f:
            self.world_map = []
            for line in reversed(f.read().splitlines()):
                self.world_map.append([int(tile) for tile in list(line.replace(' ', '0').replace('#', '1'))])   # Convert to [[int]]

    def _exec_action(self, position, action):
        x, y = position

        if action == 0:
            if self.world_map[position[1] - 1][position[0]] == 0:
                y -= 1
        if action == 1:
            if self.world_map[position[1]][position[0] + 1] == 0:
                x += 1
        if action == 2:
            if self.world_map[position[1] + 1][position[0]] == 0:
                y += 1
        if action == 3:
            if self.world_map[position[1]][position[0] - 1] == 0:
                x -= 1
        return x, y

    def _get_next_positions(self, position):
        return set(self._exec_action(position, action) for action in range(4))

    # bfs + observation calculation based on distances
    # def _calculate_observations(self):
    #     actual = (self.agt_x, self.agt_y)
    #     open_set = [actual]
    #     closed_set = set()
    #
    #     # track distances from origin
    #     self.observations = {}
    #     self.observations[actual] = 0
    #
    #     while open_set:
    #         actual = open_set.pop(0)
    #         closed_set.add(actual)
    #         for next_position in self._get_next_positions(actual):
    #             if next_position not in closed_set:
    #                 open_set.append(next_position)
    #
    #                 self.observations[next_position] = self.observations[actual] + 1

    def step(self, action):
        self.agt_x, self.agt_y = self._exec_action((self.agt_x, self.agt_y), action)

        logger.debug("{} {}".format(self.agt_x, self.agt_y))

        # create observations from agent's
        # obs = self.observations[(self.agt_x, self.agt_y)]
        # obs *= obs
        # obs = np.array([obs])
        obs = np.array([100-self.agt_x, 100-self.agt_y])

        return obs, 0, 0, None  # reset on reward

    def show(self):
        self.world_map[self.agt_y][self.agt_x] = '@'
        for line in self.world_map:
            print(''.join(line), end='')

        self.world_map[self.agt_y][self.agt_x] = ' '

    def reset(self, test=False):
        self.agt_x = self.width  // 2 - 1
        self.agt_y = self.height // 2 - 1

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# class Visualizing2DWorld(My2DWorld):
class Visualizing2DWorld(Twisted2DWorld):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.step_no = 0
        self.pos_history = []   # We need to keep position history so that we can draw the agent's movements. Although the agent has is's own buffer, it doesn't contain the x and y -- only the observation from the given point.
        self.heatmap = np.zeros((self.height, self.width))
        self.last_all_visited = self.heatmap.copy()
        self.all_visited_log = []
        self.reset_log = []  # Log every time the world resets

    def step(self, action):
        from config import CONFIG

        self.step_no += 1

        # Save current coordinates
        self.pos_history.append((self.agt_x, self.agt_y))

        # Update heat map
        self.heatmap[self.agt_y, self.agt_x] += 1

        if np.min(self.heatmap - self.last_all_visited) != 0:   # If all tiles have been visited since last record
            self.all_visited_log.append(self.step_no)
            self.last_all_visited = self.heatmap.copy()
            for coords in rewards:
                self.last_all_visited[coords] += 1     # The coords of the reward are never visited

        # Log step number
        if self.step_no % (CONFIG['AGENT']['ctor_params']['training_period'] // 20) == 0:
            logger.info("step {}".format(self.step_no))

        # Actually change the agent's position
        ret = super().step(action)

        # Remember a reset
        if ret[2] == True:
            self.reset_log.append(self.step_no)

        return ret

    def plot_map(self):
        plt.axis((0, self.width - 1, 0, self.height - 1))

        # Draw walls
        plt.imshow(np.array(self.world_map), cmap='Greys')
        # for y in range(self.height):
        #     for x in range(self.width):
        #         if self.world_map[y][x] == 1:
        #             plt.scatter(x, y, c='black', marker='s')

        # Draw rewards
        for coords in rewards:
            plt.scatter(*coords, c='green')

    def plot_actions(self):
        plt.axis((0, self.width - 1, 0, self.height - 1))

        # Draw steps
        for i in range(len(self.pos_history) - 1):
            pct = i / len(self.pos_history);

            x_old, y_old = self.pos_history[i - 1]
            x_new, y_new = self.pos_history[i]

            plt.plot([x_old, x_new], [y_old, y_new], color=(pct, 0, 1-pct))

        self.pos_history.clear()

    def plot_resets(self):
        diffs = [s2 - s1 for s1, s2 in zip(self.reset_log[:-1], self.reset_log[1:])]

        plt.xlim(0, len(diffs))
        plt.plot(range(len(diffs)), diffs, c='blue')
        plt.title('steps between restarts')

    def plot_all_visited(self):
        diffs = [s2 - s1 for s1, s2 in zip(self.all_visited_log[:-1], self.all_visited_log[1:])]

        plt.xlim(0, len(diffs))
        plt.plot(range(len(diffs)), diffs, c='blue')
        plt.title('steps between all fields being visited')

    def plot_heatmap(self):
        # htmp = self.heatmap.copy()
        # htmp /= np.max(htmp)        # make all values between 0 and 1

        hmap = plt.imshow(self.heatmap, cmap='Greys', origin='lower', norm=Normalize(vmin=0, vmax=np.max(self.heatmap)))
        plt.title('heatmap')
        plt.colorbar(hmap)
