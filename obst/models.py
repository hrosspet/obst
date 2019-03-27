from keras.models import Model
from keras.layers import Input, Dense, Concatenate

INPUT_DIM = 5
REPRE_DIMS = [32, 16]
HIDDEN_DIM = 16

SIM_OUTPUT_DIM = 1
WM_OUTPUT_DIM = INPUT_DIM
REWARD_OUTPUT_DIM = 1

def apply_shared(shared, inp):
    res = inp
    for layer in shared:
        res = layer(res)
    return res


def create_repre(layer_dims):
    repre = [Dense(layer_dim, activation='relu') for layer_dim in layer_dims]
    return repre


def create_similarity_model(shared_repre_layers, inputs):
    # connect the shared layers to their respective inputs
    branches = [apply_shared(shared_repre_layers, inp) for inp in inputs]
    output = Concatenate()(branches)
    output = Dense(HIDDEN_DIM, activation='relu')(output)
    output = Dense(SIM_OUTPUT_DIM, activation='relu', name='similarity')(output)

    similarity_model = Model(inputs=inputs, outputs=output)
    similarity_model.compile(loss='mse',
                  optimizer='rmsprop')
    similarity_model.summary()

    return similarity_model


def create_world_model(shared_repre_layers, input_common):
    representation = apply_shared(shared_repre_layers, input_common)
    world_model_output = Dense(WM_OUTPUT_DIM, activation='relu', name='world_model')(representation)
    world_model = Model(inputs=input_common, outputs=world_model_output)
    world_model.compile(loss='mse',
                  optimizer='rmsprop')
    world_model.summary()

    return world_model


def create_reward_model(shared_repre_layers, input_common):
    representation = apply_shared(shared_repre_layers, input_common)
    reward_model_output = Dense(REWARD_OUTPUT_DIM, activation='relu', name='reward_prediction')(representation)
    reward_model = Model(inputs=input_common, outputs=reward_model_output)
    reward_model.compile(loss='mse',
                  optimizer='rmsprop')
    reward_model.summary()

    return reward_model


def create_triple_models():
    input_common = Input(shape=(INPUT_DIM,), name='observation')
    input_sim = Input(shape=(INPUT_DIM,), name='observation_sim')

    # create layers for shared representation
    shared_repre_layers = create_repre(REPRE_DIMS)

    # create similarity model from shared repre
    similarity_model = create_similarity_model(shared_repre_layers, [input_common, input_sim])

    # create world model from shared repre
    world_model = create_world_model(shared_repre_layers, input_common)

    # create reward model from shared repre
    reward_model = create_reward_model(shared_repre_layers, input_common)

    return similarity_model, world_model, reward_model

