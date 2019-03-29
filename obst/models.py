import numpy as np
import random
import logging

from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout, LSTM

class SimModel():
    def __init__(self, prep_layers, obs_size=None, repr_size=None):
        obs_a = Input(shape=(obs_size,))
        obs_b = Input(shape=(obs_size,))
        prep_a  = prep_layers(obs_a)
        prep_b  = prep_layers(obs_b)

        self.sim_layers  = SimModel.create_layers(repr_size=repr_size)([prep_a, prep_b])

        self.model = Model(inputs=[obs_a, obs_b], outputs=self.sim_layers)
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    @staticmethod
    def create_layers(repr_size=None):
        repr_a = Input(shape=(repr_size,))
        repr_b = Input(shape=(repr_size,))

        outputs = concatenate([repr_a, repr_b])
        outputs = Dense(repr_size // 2, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)

        return Model(inputs=[repr_a, repr_b], outputs=outputs)

    def sim_obs(self, obs_a, obs_b): # -> float (similarity)
        return self.model.predict([np.array([obs_a]), np.array([obs_b])])[0][0]

    def sim_reprs(self, obs_a, obs_b): # -> float (similarity)
        return self.sim_layers.predict([np.array([obs_a]), np.array([obs_b])])[0][0]

    def gen_train_data(self, buffer, batch_size=None):
        half_batch = batch_size // 2
        idx = np.random.randint(0, len(buffer) - 1, 3 * half_batch)
        similar_idx = idx[:half_batch]
        dissimilar_idx_0 = idx[half_batch:half_batch*2]
        dissimilar_idx_1 = idx[half_batch*2:]
        data = np.array([observation for observation, reward, decision in buffer])

        data_x_a = np.zeros((batch_size, data.shape[-1]))
        data_x_b = np.zeros((batch_size, data.shape[-1]))
        data_y = np.zeros((batch_size, 1))

        # fill in similar samples
        data_x_a[:half_batch, :] = data[similar_idx, :]
        data_x_b[:half_batch, :] = data[similar_idx+1, :]

        data_y[:half_batch] = 1

        # fill in dissimilar samples
        data_x_a[half_batch:, :] = data[dissimilar_idx_0, :]
        data_x_b[half_batch:, :] = data[dissimilar_idx_1, :]

        return (data_x_a, data_x_a), data_y

    def nice_train(buffer, batch_size):
        (x_repr_a, x_repr_b), y_sim = self.gen_train_data(buffer, batch_size)
        self.model.train([x_repr_a, x_repr_b], y_sim)

class WMModel():
    def __init__(self, prep_layers, obs_size=None, repr_size=None):
        start_obs = Input(shape=(obs_size,), name='start_obs')
        self.prep_layers = prep_layers(start_obs)   # layers that preprocess the start observation.

        action = Input(shape=(1,), name='action')

        self.wm_layers = WMModel.create_layers(repr_size=repr_size)([self.prep_layers, action])#({'start_repr': self.prep_layers, 'wm_action': action})
        self.model = Model(inputs=[start_obs, action], outputs=self.wm_layers)
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    @staticmethod
    def create_layers(repr_size=None):
        start_repr = Input(shape=(repr_size,), name='start_repr')
        action = Input(shape=(1,), name='wm_action')

        outputs = concatenate([start_repr, action])
        outputs = Dense(repr_size, activation='relu')(outputs)
        outputs = Dense(repr_size, activation='relu')(outputs)

        return Model(inputs=[start_repr, action], outputs=outputs)

    def nice_predict(self, start_obs, action): # -> representation of resulting obs
        return self.model.predict({'start_obs': np.array([start_st]), 'action': np.array([action])})[0]

    def gen_train_data(self, buffer, batch_size=None):
        start_idx = np.random.randint(0, len(buffer) - 1, size=batch_size)

        np_buffer = np.array(buffer)
        data_x_obs = np.zeros((batch_size, buffer[0][0].shape))
        data_x_act  = np.zeros((batch_size, 1))

        # select start_st
        data_x_obs = np_buffer[0, start_idx, :]
        # select action
        data_x_act = np_buffer[2, start_idx]     # buffer: [(observation, reward, action), ...]

        # select end_st
        data_y = np_buffer[0, start_idx+1, :]   # Although we get an observation for input, we only generate the inner representation of the outcome observation, so we need to process the outcome observations beforehand. obs + action -> end_repr
        data_y = self.prep_layers.predict(data_y)

        return (data_x_obs, data_x_act), data_y

    def nice_train(buffer, batch_size):
        (x_obs, x_action), y_repr = self.gen_train_data(buffer, batch_size)
        self.model.train({'start_obs': x_obs, 'action': x_action}, y_repr)

class RewardModel():
    def __init__(self, prep_layers, obs_size=None, repr_size=None):
        input_obs = Input(shape=(obs_size,))
        self.prep_layers = prep_layers(input_obs)
        self.rew_layers  = RewardModel.create_layers(repr_size=repr_size)(self.prep_layers)

        self.model = Model(inputs=input_obs, outputs=self.rew_layers)
        self.model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    @staticmethod
    def create_layers(repr_size=None):
        inputs  = Input(shape=(repr_size,))

        outputs = Dense(repr_size, activation='relu')(inputs)
        outputs = Dense(1, activation='relu')(outputs)

        return Model(inputs=inputs, outputs=outputs)

    def predict(self, observation): # -> float (reward)
        return self.model.predict(np.array([observation]))[0]

    def gen_train_data(self, buffer, batch_size=None):
        # Get all the positions in the buffer that have a reward and find just as many that don't
        reward_records = [x for x in buffer if x[1] != 0]   # buffer: [(observation, reward, action), (obs...
        noreward_records = [random.choice(filter(lambda x: x[1] == 0), buffer) for _ in range(len(reward_records))]

        data_x = np.zeros((len(reward_records) * 2, reward_records[0][0].shape[-1]))    # <- shape of observation
        data_y = np.zeros((len(reward_records) * 2, 1))

        data_x[:len(reward_records)] = [obs for obs, rew, act in reward_records]
        data_y[:len(reward_records)] = 1

        data_x[len(reward_records):] = [obs for obs, rew, act in noreward_records]
        data_y[len(reward_records):] = 0

        return data_x, data_y

    def nice_train(buffer, batch_size):
        data_x, data_y = self.gen_train_data(buffer, batch_size)
        self.model.train(data_x, data_y)

#

class PreprocessModel():
    @staticmethod
    def create_layers(obs_size=None, repr_size=None):
        inputs  = Input(shape=(obs_size,))

        outputs = Dense((obs_size + repr_size) // 2, activation='relu')(inputs)
        outputs = Dense(repr_size, activation='relu')(outputs)

        return Model(inputs=inputs, outputs=outputs)
