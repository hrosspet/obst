from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AbstractAgent(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def behave(self, observation):
        pass

    @abstractmethod
    def reset(self):
        pass


class BufferedAgent(AbstractAgent):
    def __init__(self, buffer_size, training_period, n_actions):
        self.buffer_size = buffer_size
        self.buffer = []

        self.training_period = training_period
        self.step = 0

        self.n_actions = n_actions

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def decide(self, observation):
        pass

    def reset(self):
        self.buffer = []
        self.step = 0

    def observe(self, observation):
        self.step += 1

        self.buffer.append(observation)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if self.step % self.training_period == 0:
            self.train()

    def behave(self, observation):
        self.observe(observation)
        return self.decide(observation)


class RandomBufferedAgent(BufferedAgent):
    def decide(self, observation):
        return np.random.randint(self.n_actions)


from keras.models import Model
from keras.layers import Input, Dense, Dropout

OUTPUT_DIM = 1

def create_model(input_dim, n_layers=1, learning_rate=1e-3):
    inputs = Input(shape=(input_dim * 2,))

    outputs = inputs

    for i in range(n_layers):
        outputs = Dropout(0.5)(outputs)
        outputs = Dense(OUTPUT_DIM, activation='relu')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['acc'])
    model.summary()
    return model

class RandomBufferedKerasAgent(RandomBufferedAgent):
    def __init__(self, buffer_size, training_period, n_actions, input_dim, batch_size, steps_per_epoch, epochs, n_layers=None, lr=None):
        super().__init__(buffer_size, training_period, n_actions)

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.model = create_model(input_dim, n_layers, lr)

    def get_data_generator(self):
        # make a training batch: batch_size/2 x pair of similar inputs and batch_size/2 x pair of dissimilar inputs
        while True:
            half_batch = self.batch_size // 2
            idx = np.random.randint(0, len(self.buffer) - 1, 3 * half_batch)
            similar_idx = idx[:half_batch]
            dissimilar_idx_0 = idx[half_batch:half_batch*2]
            dissimilar_idx_1 = idx[half_batch*2:]
            data = np.array(self.buffer)

            data_x = np.zeros((self.batch_size, self.input_dim * 2))
            data_y = np.zeros((self.batch_size, 1))

            # fill in similar samples
            data_x[:half_batch, :self.input_dim] = data[similar_idx, :]
            data_x[:half_batch, self.input_dim:] = data[similar_idx+1, :]

            data_y[:half_batch] = 1

            # fill in dissimilar samples
            data_x[half_batch:, :self.input_dim] = data[dissimilar_idx_0, :]
            data_x[half_batch:, self.input_dim:] = data[dissimilar_idx_1, :]

            yield data_x, data_y


    def train(self):
        logger.debug('Agent training...')
        self.model.fit_generator(
            self.get_data_generator(),
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs
        )
        #

    def get_data_test(self):
        # make a training batch: batch_size/2 x pair of similar inputs and batch_size/2 x pair of dissimilar inputs
        data = np.array(self.buffer)
        half_batch = data.shape[0] // 2

        similar_idx = np.arange(0, data.shape[0], 2)

        idx = np.random.randint(0, data.shape[0], data.shape[0])
        dissimilar_idx_0 = idx[:half_batch]
        dissimilar_idx_1 = idx[half_batch:]

        data_x = np.zeros((data.shape[0], self.input_dim * 2))
        data_y = np.zeros((data.shape[0], 1))

        # fill in similar samples
        data_x[:half_batch, :self.input_dim] = data[similar_idx, :]
        data_x[:half_batch, self.input_dim:] = data[similar_idx+1, :]

        data_y[:half_batch] = 1

        # fill in dissimilar samples
        data_x[half_batch:, :self.input_dim] = data[dissimilar_idx_0, :]
        data_x[half_batch:, self.input_dim:] = data[dissimilar_idx_1, :]

        return data_x, data_y

    def eval(self):
        return self.model.evaluate(*self.get_data_test(), batch_size=len(self.buffer))

