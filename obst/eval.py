import matplotlib.pyplot as plt
from datetime import datetime
import logging
import numpy as np

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

            # Plot the agent's movements if it's time
            if self.world.__class__.__name__ == 'Visualizing2DWorld' and step % self.intervals['visualize'] == 0:
                plt.figure(figsize=(16, 4.8))
                self.world.plot()
                plt.savefig('logs/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_steps_' + str(step - self.intervals['visualize']) + '_' + str(step) + '.png')
                plt.close()

            if step % (self.intervals['visualize'] // 20) == 0:
                logger.info("step {}".format(step))

            # Get agent's action based on the world observation
            action = self.agent.behave(observation, reward)

            if done:
                logger.debug('reset')
                self.reset(False)
