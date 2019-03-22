from abc import ABC, abstractmethod
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

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
        self.buffer = []
        self.step = 0

    def observe(self, observation):
        self.step += 1

        if self.step % self.training_period == 0:
            self.train()

    def record(self, observation, reward, decision):
        self.buffer.append((observation, reward, decision))     # Append a tuple of the observation, the reward of the observed state, and the decision that was made on it

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def behave(self, observation, reward):
        self.observe(observation)
        decision = self.decide(observation, reward)

        self.record(observation, reward, decision)

        return decision

class RandomBufferedAgent(BufferedAgent):
    def decide(self, observation, reward):
        # We have to make the movement biased towards one side to prevent the agent from going back and forth in a small area
        return (0 if random.randint(0,6) <= 3  else 1)  # This just learns the effect of random movements and does not make any computed decissions


from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM

class RandomBufferedKerasAgent(RandomBufferedAgent):
    def __init__(self, buffer_size, training_period, n_actions, input_dim, batch_size, steps_per_epoch, epochs, n_layers=None, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = self.create_model(input_dim, n_layers, lr)

    def create_model(self, input_dim, n_layers=1, learning_rate=1e-3):
        inputs = Input(shape=(input_dim,))
        outputs = inputs

        OUTPUT_DIM = 1

        # for i in range(n_layers):
        #     outputs = Dropout(0.5)(outputs)
        outputs = Dense(input_dim, activation='relu')(outputs)
        #
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

            data_x = np.zeros((self.batch_size, self.input_dim))
            data_y = np.zeros((self.batch_size, 1))

            # fill in similar samples
            data_x[:half_batch] = data[similar_idx, :] - data[similar_idx+1, :]

            data_y[:half_batch] = 1

            # fill in dissimilar samples
            data_x[half_batch:] = data[dissimilar_idx_0, :] - data[dissimilar_idx_1, :]

            yield data_x, data_y

    def train(self):
        logger.debug('Agent training...')
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

class WorldModelBufferedKerasAgent(RandomBufferedAgent):
    """This agent learns to predict the observation that will be caused by an action."""

    def __init__(self, buffer_size, training_period, n_actions, input_dim, batch_size, steps_per_epoch, epochs, n_layers=None, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = self.create_model(input_dim, lr)

    def create_model(self, input_dim, learning_rate=1e-3):
        inputs  = Input(shape=(input_dim + 1,))
        outputs = inputs
        outputs = Dense(input_dim, activation='relu')(outputs)
        outputs = Dense(input_dim, activation='relu')(outputs)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc'])
        model.summary()
        return model

    def get_train_data_generator(self):
        while True:
            start_idxs = [np.random.randint(0, len(self.buffer)  - 1) for _ in range(self.batch_size)]

            data_x = np.zeros((self.batch_size, self.input_dim + 1))
            data_y = np.zeros((self.batch_size, self.input_dim))

            for i in range(self.batch_size):
                start_st, _, action = self.buffer[start_idxs[i]]
                end_st, _, _ = self.buffer[start_idxs[i] + 1]      # end state/observation

                data_x[i][:self.input_dim] = start_st
                data_x[i][-1] = action

                data_y[i] = end_st      # data_y is essentially data_x shifted by one

            yield data_x, data_y

    def train(self):
        logger.debug('Agent training...')
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

class RewardPredictBufferedKerasAgent(RandomBufferedAgent):
    """This agent tries to learn what observation has what reward."""

    def __init__(self, buffer_size, training_period, n_actions, input_dim, batch_size, steps_per_epoch, epochs, n_layers=None, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = self.create_model(input_dim, lr)

    def create_model(self, input_dim, learning_rate=1e-3):
        inputs  = Input(shape=(input_dim,))
        outputs = inputs
        outputs = Dense(input_dim, activation='relu')(outputs)
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
            data_x = np.zeros((self.batch_size, self.input_dim))
            data_y = np.zeros((self.batch_size, 1))

            for i in range(self.batch_size):
                observation, reward, _ = random.choice(self.buffer)

                data_x[i] = observation
                data_y[i] = reward

            yield data_x, data_y

    def train(self):
        logger.debug('Agent training...')
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


# class RewardGuideBufferedKerasAgent(BufferedAgent):
#     """Predict the reward that each of the next possible steps would give."""
#
#     def __init__(self, buffer_size, training_period, n_actions, input_dim, batch_size, steps_per_epoch, epochs, n_layers=None, lr=None):
#         super().__init__(buffer_size, training_period, n_actions)
#
#         self.input_dim = input_dim
#         self.batch_size = batch_size
#         self.steps_per_epoch = steps_per_epoch
#         self.epochs = epochs
#         self.model = self.create_model(input_dim, lr)
#
#     def create_model(self, input_dim, learning_rate=1e-3):
#         inputs  = Input(shape=(input_dim,))
#         outputs = inputs
#
#         outputs = LSTM(input_dim)(outputs)
#         outputs = Dense(input_dim, activation='relu')(outputs)
#         outputs = Dense(self.n_actions, activation='softmax')(outputs)
#
#         model = Model(inputs=inputs, outputs=outputs)
#         model.compile(loss='mse',
#                   optimizer='rmsprop',
#                   metrics=['acc'])
#         model.summary()
#         return model
#
#     def get_train_data_generator(self):       # We don't have access to the neigbouring states so we can't really train it.
#         while True:
#             np.array([observation for observation, reward, decision in self.buffer])
#
#             yield data_x, data_y
#
#     def train(self):
#         logger.debug('Agent training...')
#         self.model.fit_generator(
#             self.get_train_data_generator(),
#             steps_per_epoch=self.steps_per_epoch,
#             epochs=self.epochs
#         )
#         #
#
#     def decide(self, observation):
#         return np.argmax(self.model.predict(np.array([observation])))
#
#     def eval(self):
#         data_x, data_y = next(self.get_train_data_generator())
#         import pdb; pdb.set_trace()
#         return self.model.evaluate(data_x, data_y, batch_size=len(self.buffer))
