from abc import ABC, abstractmethod
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

import keras
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout, LSTM, GaussianNoise, LeakyReLU, BatchNormalization, Subtract, Conv2D, MaxPooling2D, Flatten
from obst.r2_score import r2_score

# Each Submodel class contains two keras models: a train_model and a use_model
# The train_model contains the shared layers so that we can train these shared layers with the model. It works on observations.
# The use_model only contains the layers made create_layers and is used for the predicting. (We make predictions from the inner representation, not straight from observations)

class Submodel(ABC):
    # train_model: Model
    # use_model: Model

    # unique_layers: Model

    def __init__(self, repr_layers, dims):
        # Layers unique to this model
        self.unique_layers = self.create_layers(dims)    # These are shared between the train_ and use_model

        # Create the two models that use them
        self.use_model   = self.create_use_model(dims)
        self.train_model = self.create_train_model(repr_layers, dims)

    @abstractmethod
    def create_use_model(self, dims) -> Model:
        pass

    @abstractmethod
    def create_train_model(self, repr_layers, dims) -> Model:
        pass

    @staticmethod
    @abstractmethod
    def create_layers(dims) -> Model:
        pass

    @abstractmethod
    def train(self, buffer, hparams) -> keras.callbacks.History:
        pass

class SimModel(Submodel):
    def __init__(self, repr_layers, dims):
        super().__init__(repr_layers, dims)

    def create_train_model(self, repr_layers, dims):
        obs_a = Input(shape=dims['obs_size'])
        obs_b = Input(shape=dims['obs_size'])
        repr_a  = repr_layers(obs_a)
        repr_b  = repr_layers(obs_b)

        sim = self.unique_layers([repr_a, repr_b])

        train_model = Model(inputs=[obs_a, obs_b], outputs=sim)
        train_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', r2_score])
        return train_model

    def create_use_model(self, dims):
        repr_a = Input(shape=(dims['repr_size'],))
        repr_b = Input(shape=(dims['repr_size'],))

        sim = self.unique_layers([repr_a, repr_b])

        return Model(inputs=[repr_a, repr_b], outputs=sim)

    @staticmethod
    def create_layers(dims):
        repr_a = Input(shape=(dims['repr_size'],))
        repr_b = Input(shape=(dims['repr_size'],))

        # outputs = concatenate([repr_a, repr_b])
        outputs = Subtract()([repr_a, repr_b])
        outputs = Dense(dims['repr_size'] // 2, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)
        # outputs = LeakyReLU(alpha=0.1)(outputs)

        return Model(inputs=[repr_a, repr_b], outputs=outputs)

    def get_train_data_gen(self, buffer, batch_size):
        while True:
            half_batch = batch_size // 2
            idx = np.random.randint(0, len(buffer) - 1, 3 * half_batch)
            similar_idx = idx[:half_batch]
            dissimilar_idx_0 = idx[half_batch:half_batch*2]
            dissimilar_idx_1 = idx[half_batch*2:half_batch*3]
            data = np.array([observation for observation, reward, decision in buffer])

            data_x_a = np.zeros((batch_size, *data.shape[1:]))
            data_x_b = np.zeros((batch_size, *data.shape[1:]))
            data_y = np.zeros((batch_size, 1))

            # fill in similar samples
            data_x_a[:half_batch, :] = data[similar_idx, :]
            data_x_b[:half_batch, :] = data[similar_idx+1, :]

            data_y[:half_batch] = 1

            # fill in dissimilar samples
            data_x_a[half_batch:, :] = data[dissimilar_idx_0, :]
            data_x_b[half_batch:, :] = data[dissimilar_idx_1, :]

            yield [data_x_a, data_x_b], data_y

    def train(self, buffer, hparams):
        logger.info("Training {}...".format(self.__class__.__name__))
        return self.train_model.fit_generator(self.get_train_data_gen(buffer, hparams['batch_size']), steps_per_epoch=hparams['steps_pe'], epochs=hparams['epochs'])

    def predict_sim(self, repr_a, repr_b):
        return self.use_model.predict([np.array([repr_a]), np.array([repr_b])])[0]

class WMModel(Submodel):
    def __init__(self, repr_layers, dims):
        super().__init__(repr_layers, dims)

    def create_train_model(self, repr_layers, dims):
        start_obs  = Input(shape=dims['obs_size'], name='start_obs')
        start_repr = repr_layers(start_obs)   # layers that preprocess the start observation.

        action = Input(shape=(1,), name='action')

        wm =self.unique_layers([start_repr, action])#({'start_repr': self.repr_layers, 'wm_action': action})

        train_model = Model(inputs=[start_obs, action], outputs=wm)
        train_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', r2_score])
        return train_model

    def create_use_model(self, dims):
        start_repr = Input(shape=(dims['repr_size'],), name='start_repr')
        action = Input(shape=(1,), name='action')

        wm =self.unique_layers([start_repr, action])#({'start_repr': self.repr_layers, 'wm_action': action})

        return Model(inputs=[start_repr, action], outputs=wm)

    @staticmethod
    def create_layers(dims):
        start_repr = Input(shape=(dims['repr_size'],), name='start_repr')
        action = Input(shape=(1,), name='action')

        outputs = concatenate([start_repr, action])
        outputs = Dense(dims['repr_size'], activation='relu')(outputs)
        outputs = Dense(dims['repr_size'])(outputs)
        outputs = LeakyReLU(alpha=0.2)(outputs)

        return Model(inputs=[start_repr, action], outputs=outputs)

    def get_train_data_gen(self, buffer, batch_size, repr_model):
        while True:
            start_idx = np.random.randint(0, len(buffer) - 1, size=batch_size)

            np_buffer = np.array(buffer)
            data_x_obs = np.zeros((batch_size, buffer[0][0].shape[0]))  # observation size
            data_x_act = np.zeros((batch_size, 1))

            # select start_st
            data_x_obs = np.stack(np_buffer[start_idx, 0])
            # select action
            data_x_act = np_buffer[start_idx, 2]     # buffer: [(observation, reward, action), ...]

            # select end_st
            data_y = np.stack(np_buffer[start_idx+1, 0])   # np.stack: numpy array of np arrays -> 2 dimensional np array
            data_y = repr_model.model.predict(data_y)       # Although we get an observation for input, we only generate the inner representation of the outcome observation, so we need to process the outcome observations beforehand. obs + action -> end_repr
            yield [data_x_obs, data_x_act], data_y

    def train(self, buffer, repr_model, hparams):    # We need the prep model so that we can generate our y data from the observations in the buffer.
        logger.info("Training {}...".format(self.__class__.__name__))
        return self.train_model.fit_generator(self.get_train_data_gen(buffer, hparams['batch_size'], repr_model), steps_per_epoch=hparams['steps_pe'], epochs=hparams['epochs'])

    def predict_wm(self, start_repr, action): # -> representation of resulting obs
        return self.use_model.predict({'start_repr': np.array([start_repr]), 'action': np.array([action])})[0]

class RewardModel(Submodel):
    def __init__(self, repr_layers, dims):
        super().__init__(repr_layers, dims)

    def create_train_model(self, repr_layers, dims):
        input_obs = Input(shape=dims['obs_size'])
        _repr = repr_layers(input_obs)

        rew = self.unique_layers(_repr)

        train_model = Model(inputs=input_obs, outputs=rew)
        train_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy', r2_score])
        return train_model

    def create_use_model(self, dims):
        _repr = Input(shape=(dims['repr_size'],))

        rew = self.unique_layers(_repr)

        return Model(inputs=_repr, outputs=rew)

    @staticmethod
    def create_layers(dims):
        inputs  = Input(shape=(dims['repr_size'],))

        outputs = Dense(dims['repr_size'], activation='relu')(inputs)
        outputs = Dense(1, activation='relu')(outputs)

        return Model(inputs=inputs, outputs=outputs)

    def get_train_data_gen(self, buffer, batch_size):
        while True:
            # Get all the positions in the buffer that have a reward and find just as many that don't
            reward_records = [x for x in buffer if x[1] != 0]   # buffer: [(observation, reward, action), (obs...
            noreward_records = [random.choice([x for x in buffer if x[1] == 0]) for _ in range(len(reward_records))]

            data_x = np.zeros((len(reward_records) * 2, reward_records[0][0].shape[-1]))    # <- shape of observation
            data_y = np.zeros((len(reward_records) * 2, 1))

            data_x[:len(reward_records)] = [obs for obs, rew, act in reward_records]
            data_y[:len(reward_records)] = 1

            data_x[len(reward_records):] = [obs for obs, rew, act in noreward_records]
            data_y[len(reward_records):] = 0

            yield data_x, data_y

    def train(self, buffer, hparams):
        print(len([x for x in buffer if x[1] != 0]))
        if len([x for x in buffer if x[1] != 0]) != 0:  # If we've actually got some rewards in buffer to train on
            logger.info("Training {}...".format(self.__class__.__name__))
            return self.train_model.fit_generator(self.get_train_data_gen(buffer, hparams['batch_size']), steps_per_epoch=hparams['steps_pe'], epochs=hparams['epochs'])

    def predict_rew(self, repres): # -> float (reward)
        return self.use_model.predict(np.array([repres]))[0]

# Models that take in an observation and generate intermediate layers
class PreprocessModel(ABC):
    def __init__(self, repr_layers, obs_size):
        # Create a simple model that does observation -> inner_representation

        input_obs = Input(shape=obs_size)
        output_repr = repr_layers(input_obs)

        self.model = Model(inputs=input_obs, outputs=output_repr)

    @staticmethod
    @abstractmethod
    def create_layers(obs_size, repr_size) -> Model:
        pass

    def get_repr(self, observation):
        return self.model.predict(np.array([observation]))[0]


class VectorPreprocessModel(PreprocessModel):
    def __init__(self, repr_layers, obs_size):
        super().__init__(repr_layers, obs_size)

    @staticmethod
    def create_layers(obs_size, repr_size):
        inputs  = Input(shape=obs_size)

        outputs = Dense(repr_size, activation='relu')(inputs)
        outputs = Dense(repr_size)(outputs)
        outputs = LeakyReLU(alpha=0.1)(outputs)
        # outputs = GaussianNoise(0.1)(outputs)

        return Model(inputs=inputs, outputs=outputs)

class ImagePreprocessModel(PreprocessModel):
    def __init__(self, repr_layers, obs_size):
        super().__init__(repr_layers, obs_size)     #

    @staticmethod
    def create_layers(obs_size, repr_size):
        inputs = outputs = Input(shape=obs_size)

        outputs = Conv2D(8, kernel_size=(12, 12), strides=(8, 8), activation='relu')(inputs)  # -> (21, 21, 8)
        outputs = Conv2D(16, (3, 3), strides=(3, 3), activation='relu')(outputs)    # -> (7, 7, 64)
        outputs = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(outputs)
        outputs = Dropout(0.25)(outputs)
        outputs = Flatten()(outputs)
        outputs = Dense(128, activation='relu')(outputs)
        outputs = Dense(repr_size)(outputs)
        outputs = LeakyReLU(alpha=0.1)(outputs)

        model=Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
