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
        self.world.reset()
        self.agent.reset()

    def eval(self, n_steps, test=False):
        self.reset(test)

        score = 0
        reward = 0
        action = 0

        for step in range(n_steps):
            # get world's reaction
            observation, reward, done, _ = self.world.step(action)

            # Plot the agent's movements if it's time
            if self.world.__class__.__name__ == 'Visualizing2DWorld' and step % self.vis_steps == 0:
                plt.figure(figsize=(16, 4.8))
                self.world.plot()
                plt.savefig('logs/' + datetime.now().strftime("%Y%m%d%H%M%S") + '_steps_' + str(step - self.vis_steps) + '_' + str(step) + '.png')
                plt.close()

            if step % (self.vis_steps // 20) == 0:
                logger.info("step {}".format(step))

            # if step % 100 == 50:
            #     import pdb; pdb.set_trace()

            # get agent's action based on the world observation
            action = self.agent.behave(observation, reward)

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
