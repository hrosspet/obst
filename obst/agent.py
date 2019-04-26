from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from keras import backend as K

from obst.models import PreprocessModel, SimModel, WMModel, RewardModel

class AgentMode(Enum):
    RANDOM  = 0
    EXPLORE = 1
    EXPLOIT = 2

class AbstractAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def behave(self, observation, reward):
        pass

    @abstractmethod
    def reset(self):
        pass


class BufferedAgent(AbstractAgent):
    def __init__(self, buffer_size, training_period, n_actions):
        self.buffer_size = buffer_size
        self.buffer = []	# Contains the n last observations that the agent acted on

        self.training_period = training_period
        self.step = 0

        self.n_actions = n_actions

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def decide(self, observation, reward):
        """Decide on an action based on an observation and reward of the current state"""
        pass

    def reset(self):
        # self.buffer.clear()   # = []    # We have to use clear() so that
        # self.step = 0
        pass

    def observe(self, observation):
        self.step += 1

        if self.step % self.training_period == 0:
            self.train()

    def record(self, observation, reward, decision):
        # Append a tuple of the observation, the reward of the observed state, and the decision that was made on it.
        # We keep the observation and not the inner representation because the network that extracts data from the observation changes over time.
        self.buffer.append((observation, reward, decision))

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def behave(self, observation, reward):
        self.observe(observation)   # trains every 10000 steps
        decision = self.decide(observation, reward)

        self.record(observation, reward, decision)

        return decision

#

class ExplorationAgent(BufferedAgent):
    def __init__(self, mode, repr_model, buffer_size, training_period, n_actions, dims, hparams):
        super().__init__(buffer_size, training_period, n_actions)

        self.mode = AgentMode.RANDOM#AgentMode[mode] # parses the string as an enum
        self.dims  = dims
        self.hparams = hparams

        # Create models
        self.repr_layers = repr_model.create_layers(dims['obs_size'], dims['repr_size'])
        self.repr_model  = repr_model(self.repr_layers, dims['obs_size'])

        self.sim_model = SimModel(self.repr_layers, dims)
        self.wm_model  = WMModel(self.repr_layers, dims)
        self.reward_model = RewardModel(self.repr_layers, dims)

    def decide(self, observation, reward):
        representation = self.repr_model.get_repr(observation)
        candidates_repr = {}     # {action, predicted_repr}

        # Predict the observation for each action
        for act_no in range(self.n_actions):
            candidates_repr[act_no] = self.wm_model.predict_wm(representation, act_no)

        if self.mode == AgentMode.EXPLORE:
            # Find the one with the lowest similarity
            candidates_sim = {}     # {action, sim}

            for action, outcome in candidates_repr.items():
                candidates_sim[action] = self.sim_model.predict_sim(representation, outcome)
                candidates_sim[action] += random.uniform(0, 0.05)   # in case all predictions are 0

            lowest_sim = min(candidates_sim, key=candidates_sim.get)

            for action, sim in candidates_sim.items():
                logger.debug('{}: {}  {}\t->\t{}\t ({}) {}'.format(action, observation, representation, candidates_repr[action], sim, '*' if action == lowest_sim else ''))

            return lowest_sim

        if self.mode == AgentMode.EXPLOIT:
            # Find the one with the highest reward
            candidates_reward = {}  # {action, predicted_reward}

            for action, outcome in candidates_repr.items():
                candidates_reward[action] = self.reward_model.predict_rew(outcome)
                candidates_reward[action] += random.uniform(0, 0.05)    # in case all predictions are 0

            highest_reward = max(candidates_reward, key=candidates_reward.get)

            for action, pred_reward in candidates_reward.items():
                logger.debug('{}: {}  {}\t->\t{}\t ({}) {}'.format(action, observation, representation, candidates_repr[action], pred_reward, '*' if action == highest_reward else ''))

            return highest_reward

        if self.mode == AgentMode.RANDOM:
            return random.randint(0, self.n_actions - 1)

    def train(self):
        # import pdb;pdb.set_trace()
        # Gets called every 10,000 steps to train the various models we're using
        self.sim_model.train(self.buffer, self.hparams)
        self.wm_model.train(self.buffer, self.repr_model, self.hparams)
        self.reward_model.train(self.buffer, self.hparams)

        if self.mode == AgentMode.RANDOM:
            logger.info('Switching from RANDOM to EXPLORE mode.')

            # self.mode = AgentMode.EXPLOIT     # Switch to exploration after the initial period of random movement
            self.training_period = 100

            self.mode = AgentMode.EXPLORE     # Switch to exploration after the initial period of random movement

    # def save_weights(self, directory):
    #     self.repr_layers.save('%/shared_layers.h5' % directory)
    #     self.sim_model.save('%/.h5' % directory)
