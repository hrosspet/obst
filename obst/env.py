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

        self.reset(False)

    def step(self, action):
        # Respond to action
        if action == 0:
            if self.agt_y > 0:
                self.agt_y -= 1
        if action == 1:
            if self.agt_x < self.width:
                self.agt_x += 1
        if action == 2:
            if self.agt_y < self.height:
                self.agt_y += 1
        if action == 3:
            if self.agt_x > 0:
                self.agt_x -= 1

        # An arbitary set of numbers that changes for each state
        obs = (math.log(self.agt_x+1), math.log(self.agt_x, self.agt_y+2), math.log(self.width - self.agt_x+1, 10), math.log(self.height - self.agt_y+1), math.log(abs(self.agt_y - self.agt_x)+1))
        return obs, 0, False, None

    def reset(self, test):
        self.agt_x = self.width  // 2
        self.agt_y = self.height // 2

from datetime import datetime
import matplotlib.pyplot as plt

VIS_STEPS = 10000
global RUN_ID

class Visualizing2DWorld(My2DWorld):
    def __init__(self, width, height):
        super().__init__(width, height)

        self.step_no = 0

        self.init_plot()

    def init_plot(self):
        self.vis = plt.subplot()    # Visualization

        self.vis.axis((0, self.width, 0, self.height))
        self.vis.grid(True)

    def step(self, action):
        # Keep track of the old coordinates before the agent moves
        old_agt_x, old_agt_y = self.agt_x, self.agt_y

        # Log the step number
        if (self.step_no % 1000 == 0): logging.info('Step {}'.format(self.step_no))

        if (self.step_no % VIS_STEPS == 0):
            plt.title('Steps {} - {}'.format(self.step_no - VIS_STEPS, self.step_no))
            plt.savefig('logs/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_steps_' + str(self.step_no - VIS_STEPS) + '_' + str(self.step_no) + '.png')
            self.init_plot()

        #
        ret = super().step(action)
        # print('Step', self.step_no, '\t', (old_agt_x, old_agt_y), action, (self.agt_x, self.agt_y))

        pct = (self.step_no % VIS_STEPS) / VIS_STEPS; pct = min(pct, 1)    # between 0 and 1 depending on how soon the stwps will be visualized
        self.vis.plot([old_agt_x, self.agt_x], [old_agt_y, self.agt_y], color=(pct, 0, 1-pct))

        self.step_no += 1

        return ret
