from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM

from obst.models import PreprocessModel, SimModel, WMModel, RewardModel

class AgentMode(Enum):
    EXPLORE = 0
    EXPLOIT = 1

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
        self.buffer.clear()   # = []    # We have to use clear() so that
        self.step = 0

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
    def __init__(self, mode, buffer_size, training_period, n_actions, obs_size, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.mode = AgentMode[mode] # parses the string as an enum
        self.repr_size = repr_size
        self.obs_size = obs_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        # Create models
        self.prep_layers = PreprocessModel.create_layers(obs_size, repr_size)
        self.prep_model  = PreprocessModel(self.prep_layers, obs_size)

        self.sim_model = SimModel(self.prep_layers, obs_size, repr_size)
        self.wm_model  = WMModel(self.prep_layers, obs_size, repr_size)
        self.reward_model = RewardModel(self.prep_layers, obs_size, repr_size)

    def decide(self, observation, reward):
        representation = self.prep_model.get_repr(observation)
        candidates_repr = {}     # {action, predicted_repr}

        # Predict the observation for each action
        for act_no in range(self.n_actions):
            candidates_repr[act_no] = self.wm_model.predict_wm(representation, act_no)

        if self.mode == AgentMode.EXPLORE:
            # Find the one with the lowest similarity
            candidates_sim = {}     # {action, sim}

            for action, outcome in candidates_repr.items():
                candidates_sim[action] = self.sim_model.predict_sim(representation, outcome)

            lowest_sim = min(candidates_sim, key=candidates_sim.get)

            if all(value == 0 for value in candidates_sim.values()):     # If they're all 0s then chose a random action so we at least get some useful training data
                lowest_sim = random.randint(0, self.n_actions - 1)

            for action, sim in candidates_sim.items():
                logger.debug('{}: {}\t->\t{}\t ({}) {}'.format(action, observation, candidates_repr[action], sim, '*' if action == lowest_sim else ''))

            return lowest_sim

        elif self.mode == AgentMode.EXPLOIT:
            # Find the one with the highest reward
            candidates_reward = {}  # {action, predicted_reward}

            for action, outcome in candidates_repr.items():
                candidates_reward[action] = self.reward_agent.predict(observation)

            highest_reward = max(candidates_reward, key=candidates_reward.get)

            if all(value == 0 for value in candidates_reward.values()):     # If they're all 0s then chose a random action so we at least get some useful training data
                highest_reward = random.randint(0, self.n_actions - 1)

            for action, pred_reward in candidates_reward.items():
                logger.debug('{}: {}\t->\t{}\t ({}) {}'.format(action, observation, candidates_repr[action], pred_reward, '*' if action == highest_reward else ''))

            return highest_reward

    def train(self):
        # Gets called every 10,000 steps to train the various models we're using
        self.sim_model.train(self.buffer, self.batch_size, epochs=self.epochs, steps_pe=self.steps_per_epoch)
        self.wm_model.train(self.buffer, self.batch_size, self.prep_model, epochs=self.epochs, steps_pe=self.steps_per_epoch)
        self.reward_model.train(self.buffer, self.batch_size, epochs=self.epochs, steps_pe=self.steps_per_epoch)
