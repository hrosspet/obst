import matplotlib.pyplot as plt
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Eval():
    def __init__(self, world, agent, training_steps, test_steps, vis_steps):
        self.world = world
        self.agent = agent
        self.training_steps = training_steps
        self.test_steps = test_steps
        self.vis_steps = vis_steps

        self.training_score = 0
        self.test_score = 0

    def reset(self, test):
        self.world.reset(test)
        self.agent.reset()

    def eval(self, n_steps, test=False):
        self.reset(test)

        score = 0

        observation = np.zeros(5)#self.world.state.observation
        reward      = 0#self.world.state.reward

        for i in range(n_steps):
            # Plot the agent's movements if it's time
            if i % self.vis_steps == 0:
                self.world.plot_actions()
                plt.savefig('logs/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_steps_' + str(i - self.vis_steps) + '_' + str(i) + '.png')
                plt.close()

            if i % (self.vis_steps // 5) == 0:
                logger.info("step {}".format(i))

            # get agent's action based on the world observation
            action = self.agent.behave(observation, reward)

            # get world's reaction
            observation, reward, done, _ = self.world.step(action)

            score += reward

            if done:
                logger.debug('reset')
                self.reset(False)

        return score

    def train(self):
        self.training_score = self.eval(self.training_steps)
        return self.training_score / self.training_steps

    def test(self):
        self.test_score = self.eval(self.test_steps, test=True)
        logger.info('test results:')
        logger.info(self.agent.eval())
        return self.test_score / self.test_steps
