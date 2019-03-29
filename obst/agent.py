from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

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

class RandomBufferedAgent(BufferedAgent):
    def decide(self, observation, reward):
        # We have to make the movement biased towards one side to prevent the agent from going back and forth in a small area
        return (0 if random.randint(0,6) <= 3  else 1)  # This just learns the effect of random movements and does not make any computed decissions

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM

from obst.models import PreprocessModel, SimModel, WMModel, RewardModel

class SimBufferedKerasAgent(RandomBufferedAgent):
    def __init__(self, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.repr_size = repr_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = SimBufferedKerasAgent.create_model(repr_size, lr)

    @staticmethod
    def create_model(repr_size, learning_rate=1e-3):
        inputs = Input(shape=(repr_size,))
        outputs = inputs

        OUTPUT_DIM = 1

        outputs = Dense(repr_size, activation='relu')(outputs)
        outputs = Dense(OUTPUT_DIM, activation='relu')(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc'])
        model.summary()
        return model

    def get_train_data_generator(self):
        # make a training batch: batch_size/2 x pair of similar inputs and batch_size/2 x pair of dissimilar inputs
        while True:
            half_batch = self.batch_size // 2
            idx = np.random.randint(0, len(self.buffer) - 1, 3 * half_batch)
            similar_idx = idx[:half_batch]
            dissimilar_idx_0 = idx[half_batch:half_batch*2]
            dissimilar_idx_1 = idx[half_batch*2:]
            data = np.array([observation for observation, reward, decision in self.buffer])

            data_x = np.zeros((self.batch_size, self.repr_size))
            data_y = np.zeros((self.batch_size, 1))

            # fill in similar samples
            data_x[:half_batch, :self.repr_size] = data[similar_idx, :]
            data_x[:half_batch, self.repr_size:] = data[similar_idx+1, :]

            data_y[:half_batch] = 1

            # fill in dissimilar samples
            data_x[half_batch:, :self.repr_size] = data[dissimilar_idx_0, :]
            data_x[half_batch:, self.repr_size:] = data[dissimilar_idx_1, :]

            yield data_x, data_y

    def train(self):
        logger.info('Training {}...'.format(self.__class__.__name__))
        self.model.fit_generator(
            self.get_train_data_generator(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        #

    def eval(self):
        import pdb; pdb.set_trace()
        data_x, data_y = next(self.get_train_data_generator())
        return self.model.evaluate(data_x, data_y, batch_size=len(self.buffer))

    def predict(self, obs_a, obs_b): # -> float (similarity)
        inp = np.zeros(self.repr_size)
        return self.model.predict(np.array([obs_a - obs_b]))[0][0]

class WorldModelBufferedKerasAgent(RandomBufferedAgent):
    """This agent learns to predict the observation that will be caused by an action."""

    def __init__(self, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.repr_size = repr_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = WorldModelBufferedKerasAgent.create_model(repr_size, lr)

    @staticmethod
    def create_model(repr_size, learning_rate=1e-3):
        inputs  = Input(shape=(repr_size + 1,))
        outputs = inputs
        outputs = Dense(repr_size, activation='relu')(outputs)
        outputs = Dropout(0.4)(outputs)
        outputs = Dense(repr_size, activation='relu')(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc'])
        model.summary()
        return model

    def get_train_data_generator(self):
        while True:
            start_idxs = [np.random.randint(0, len(self.buffer)  - 1) for _ in range(self.batch_size)]

            data_x = np.zeros((self.batch_size, self.repr_size + 1))
            data_y = np.zeros((self.batch_size, self.repr_size))

            for i in range(self.batch_size):
                start_st, _, action = self.buffer[start_idxs[i]]
                end_st, _, _ = self.buffer[start_idxs[i] + 1]      # end state/observation

                data_x[i][:self.repr_size] = start_st
                data_x[i][-1] = action

                data_y[i] = end_st      # data_y is essentially data_x shifted by one

            yield data_x, data_y

    def train(self):
        logger.info('Training {}...'.format(self.__class__.__name__))
        self.model.fit_generator(
            self.get_train_data_generator(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        #

    def eval(self):
        data_x, data_y = next(self.get_train_data_generator())
        import pdb; pdb.set_trace()
        return self.model.evaluate(data_x, data_y, batch_size=len(self.buffer))

    def predict(self, start_st, action): # -> resulting observation
        data_x = np.zeros((1, self.repr_size + 1))
        data_x[0][:self.repr_size] = start_st
        data_x[0][-1:] = action

        return self.model.predict(data_x)[0]


class RewardPredictBufferedKerasAgent(RandomBufferedAgent):
    """This agent tries to learn what observation has what reward."""

    def __init__(self, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.repr_size = repr_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = RewardPredictBufferedKerasAgent.create_model(repr_size, lr)

    @staticmethod
    def create_model(repr_size, learning_rate=1e-3):
        inputs  = Input(shape=(repr_size,))
        outputs = inputs
        outputs = Dense(repr_size, activation='relu')(outputs)
        # outputs = Dropout(0.5)(outputs)
        outputs = Dense(1, activation='relu')(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc'])
        model.summary()
        return model

    def get_train_data_generator(self):
        while True:
            data_x = np.zeros((self.batch_size, self.repr_size))
            data_y = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                observation, reward, _ = random.choice(self.buffer)

                data_x[i] = observation
                data_y[i] = reward

            yield data_x, data_y

    def train(self):
        logger.info('Training {}...'.format(self.__class__.__name__))
        self.model.fit_generator(
            self.get_train_data_generator(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        #

    def eval(self):
        data_x, data_y = next(self.get_train_data_generator())
        import pdb; pdb.set_trace()
        return self.model.evaluate(data_x, data_y, batch_size=len(self.buffer))

    def predict(self, observation): # -> float (reward)
        return self.model.predict(np.array([observation]))[0]

class ExplorationAgent(BufferedAgent):
    def __init__(self, mode, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.repr_size = repr_size
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs

        self.wm_agent     = WorldModelBufferedKerasAgent(buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=lr)
        self.reward_agent = RewardPredictBufferedKerasAgent(buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=lr)
        self.sim_agent    = SimBufferedKerasAgent(buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=lr)

        # Make them all use the same buffer. We do the recording for them.
        self.wm_agent.buffer     = self.buffer
        self.reward_agent.buffer = self.buffer
        self.sim_agent.buffer    = self.buffer

    def train(self):
        # Gets called every 10,000 steps to train the sub-agents that we're using
        self.wm_agent.train()
        self.reward_agent.train()
        self.sim_agent.train()

class CuriousExplorationAgent(ExplorationAgent):
    def __init__(self, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr)

    def decide(self, observation, reward):
        candidates_obs = {}     # {action, predicted_observation}

        # Predict the observation for each action
        for act_no in range(self.n_actions):
            candidates_obs[act_no] = self.wm_agent.predict(observation, act_no)

        # Find the one with the highest reward
        candidates_reward = {}

        for action, outcome in candidates_obs.items():
            candidates_reward[action] = self.reward_agent.predict(observation)

        highest_reward = max(candidates_reward, key=candidates_reward.get)

        if all(value == 0 for value in candidates_reward.values()):     # If they're all 0s then chose a random action so we at least get some useful training data
            highest_reward = random.randint(0, self.n_actions - 1)

        for action, pred_reward in candidates_reward.items():
            print('{}: {}\t->\t{}\t ({}) {}'.format(action, observation, candidates_obs[action], pred_reward, '*' if action == highest_reward else ''))

        return highest_reward

class RewardExplorationAgent(ExplorationAgent):
    def __init__(self, buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr=None):
        super().__init__(buffer_size, training_period, n_actions, repr_size, batch_size, steps_per_epoch, epochs, lr)

    def decide(self, observation, reward):
        candidates_obs = {}     # {action, predicted_observation}

        # Predict the observation for each action
        for act_no in range(self.n_actions):
            candidates_obs[act_no] = self.wm_agent.predict(observation, act_no)

        # Find the one with the lowest similarity
        candidates_sim = {}

        for action, outcome in candidates_obs.items():
            candidates_sim[action] = self.sim_agent.predict(observation, outcome)

        lowest_sim = min(candidates_sim, key=candidates_sim.get)

        if all(value == 0 for value in candidates_sim.values()):     # If they're all 0s then chose a random action so we at least get some useful training data
            lowest_sim = random.randint(0, self.n_actions - 1)

        for action, sim in candidates_sim.items():
            print('{}: {}\t->\t{}\t ({}) {}'.format(action, observation, candidates_obs[action], sim, '*' if action == lowest_sim else ''))

        return lowest_sim

class NewExplorationAgent(BufferedAgent):
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
