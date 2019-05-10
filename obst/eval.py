import matplotlib.pyplot as plt
from datetime import datetime
import logging
import numpy as np

from obst.config import CONFIG
from obst.env import Visualizing2DWorld

logger = logging.getLogger(__name__)

class Eval():
    def __init__(self, world, agent, intervals):
        self.world = world
        self.agent = agent
        self.intervals = intervals

        self.training_score = 0
        self.test_score = 0

    def reset(self, test):
        self.world.reset()
        self.agent.reset()

    def eval(self, n_steps, test=False):
        self.reset(test)

        reward = 0
        action = 0

        for step in range(n_steps):
            # get world's reaction
            observation, reward, done, _ = self.world.step(action)

            # Get agent's action based on the world observation
            action = self.agent.behave(observation, reward)

            if isinstance(self.world, Visualizing2DWorld) and step % CONFIG['INTERVALS']['visualize'] == 0:
                plt.figure(figsize=(15, 10))

                plt.subplot(2, 3, 2)
                self.world.plot_map()
                self.world.plot_actions()
                plt.subplot(2, 3, 4)
                self.world.plot_resets()
                plt.subplot(2, 3, 5)
                self.world.plot_all_visited()
                plt.subplot(2, 3, 1)
                self.world.plot_heatmap()

                plt.subplot(2, 3, 3)
                plt.imshow(np.array(self.world.world_map), cmap='Greys')
                self.agent.plot_sim_map(self.world.width, self.world.height, self.world.agt_x, self.world.agt_y)

                plt.savefig('logs/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_steps_' + str(step - CONFIG['INTERVALS']['visualize']) + '_' + str(step) + '.png')
                plt.close()

            if done:
                logger.debug('reset')
                self.reset(False)
