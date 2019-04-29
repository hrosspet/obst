from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List
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
    def __init__(self, mode, repr_model, buffer_size, training_period, n_actions, tree_depth, dims, hparams):
        super().__init__(buffer_size, training_period, n_actions)

        self.mode = AgentMode.RANDOM#AgentMode[mode] # parses the string as an enum
        self.dims  = dims
        self.hparams = hparams
        self.tree_depth = tree_depth

        # Create models
        self.repr_layers = repr_model.create_layers(dims['obs_size'], dims['repr_size'])
        self.repr_model  = repr_model(self.repr_layers, dims['obs_size'])

        self.sim_model = SimModel(self.repr_layers, dims)
        self.wm_model  = WMModel(self.repr_layers, dims)
        self.reward_model = RewardModel(self.repr_layers, dims)

        self.step_plan = []

    def decide(self, observation, reward):      # Just a wrapper for debug
        current=self.repr_model.get_repr(observation)
        logger.debug('current: {}'.format(current))
        act=self.decide_(observation, reward)
        logger.debug('current: {} + {} -> \t{}'.format(current, act, self.wm_model.predict_wm(current, act)))
        return act

    def decide_(self, observation, reward):
        current_repr = self.repr_model.get_repr(observation)    # Repr of our current state

        if self.mode == AgentMode.RANDOM: return random.randint(0, self.n_actions - 1)

        # If we've already planned some steps ahead, stick to the plan. Otherwise go on to generate steps now.
        if len(self.step_plan) != 0: return self.step_plan.pop(0)

        # Create a tree of potential states
        current_state = PotentialState(repr=current_repr, sim=1, reward=reward,     # Not really potential but yeah
                        models=(self.wm_model, self.sim_model, self.reward_model))
        current_state.predict_children(self.tree_depth, self.n_actions)

        # Now find the actions we want to take depending on the current criteria
        if self.mode == AgentMode.EXPLORE:
            # Find the one with the lowest similarity
            def find_action_with_lowest_sim(state, actions: List[int]) -> (List[int], int):    # actions: list of actions taken to get from here to this state
                if len(state.children) == 0: return [], 1

                # The best node so far is the one we're currently at. Once we search though its children though, we'll probably find a better one.
                lowest_sim = state.sim  # Lowest sim found
                lowest_actions = actions # Actions to get to it

                for action, outcome in state.children.items():
                    # Find if it has a child that's better
                    outcome_lowest_actions, outcome_lowest_sim = find_action_with_lowest_sim(outcome, actions + [action])
                    if outcome_lowest_sim < lowest_sim:
                        lowest_sim = outcome_lowest_sim
                        lowest_actions = outcome_lowest_actions

                return lowest_actions, lowest_sim

            self.step_plan, planned_sim = find_action_with_lowest_sim(current_state, [])

            logger.debug('actions: {} -> {}'.format(self.step_plan, planned_sim))

            return self.step_plan.pop(0)    # We only ever take the first step towards the desired state. But unless a better one is found next time, the next step will be taken any way.

        if self.mode == AgentMode.EXPLOIT:
            # Find the one with the highest reward
            def find_action_with_highest_reward(state, actions: List[int]):    # actions: list of actions taken to get from here to this state
                if len(state.children) == 0: return

                # The best node so far is the one we're currently at. Once we search though its children though, we'll probably find a better one.
                highest_reward = state.sim  # Lowest sim found
                highest_actions = actions # Actions to get to it

                for action, outcome in state.children.items():
                    # Find if it has a child that's better
                    outcome_highest_actions, outcome_highest_reward = find_action_with_highest_reward(outcome, actions + [action])
                    if outcome_highest_reward < highest_reward:
                        highest_reward = outcome_highest_reward
                        highest_actions = outcome_highest_actions

                return highest_actions, highest_reward

            find_action_with_highest_reward(current_state, [])

            return planned_actions[0]    # We only ever take the first step towards the desired state. But unless a better one is found next time, the next step will be taken any way.

    def train(self):
        # import pdb;pdb.set_trace()
        # Gets called every 10,000 steps to train the various models we're using
        self.sim_model.train(self.buffer, self.hparams)
        self.wm_model.train(self.buffer, self.repr_model, self.hparams)
        # self.reward_model.train(self.buffer, self.hparams)
        logger.info('Reward model training disabled')

        if self.mode == AgentMode.RANDOM:
            logger.info('Switching from RANDOM to EXPLORE mode.')

            # self.mode = AgentMode.EXPLOIT     # Switch to exploration after the initial period of random movement
            self.training_period = 100

            self.mode = AgentMode.EXPLORE     # Switch to exploration after the initial period of random movement

class PotentialState:
    def __init__(self, repr, models, sim=None, reward=None, parent_repr=None):   # parent_repr used only when sim isn't set to predict similarity with previous state
        self.wm_model, self.sim_model, self.reward_model = models    # We have to store references to the models that we use to guess stuff

        self.repr   = repr    # Predicted representation
        self.sim    = sim    if sim    else self.sim_model.predict_sim(parent_repr, self.repr) + random.uniform(0, 0.05)
        self.reward = reward if reward else self.reward_model.predict_rew(self.repr)  # Predict automatically if not specified.

        self.children = {}  # To be computed later

    def predict_children(self, levels, n_actions):
        if levels > 0:
            # Predict the state resulting from each action
            for act_no in range(n_actions):
                child_repr = self.wm_model.predict_wm(self.repr, act_no)
                self.children[act_no] = PotentialState(child_repr, parent_repr=self.repr, models=(self.wm_model, self.sim_model, self.reward_model))
                logger.debug((4-levels)*4*' ' + str(self.children[act_no]))
                self.children[act_no].predict_children(levels - 1, n_actions)

    def __repr__(self):
        return 'repr: {}\tsim: {}\treward: {}'.format(self.repr, self.sim, self.reward)
