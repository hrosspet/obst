from obst.env import OneHot1DWorld, OneHot1DCyclicWorld, My2DWorld, Visualizing2DWorld
from obst.agent import ExplorationAgent, RandomBufferedKerasAgent, WorldModelBufferedKerasAgent, RewardPredictBufferedKerasAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
        'CONSTRUCTOR': Visualizing2DWorld,
        'PARAMS': {
            # 'size': 1000,
            'width': 50, 'height': 50,
        }
    },
    'AGENT': {
        # 'CONSTRUCTOR': RandomBufferedKerasAgent,
        # 'CONSTRUCTOR': WorldModelBufferedKerasAgent,
        # 'CONSTRUCTOR': RewardPredictBufferedKerasAgent,
        'CONSTRUCTOR': ExplorationAgent,
        'PARAMS': {
            'buffer_size': 10000,
            'training_period': 10000,
            'n_actions': 2,
            'input_dim': 5,     # size of observation
            'batch_size': 32,
            'steps_per_epoch': 1000,
            'epochs': 2,
            'lr': 1e-3,
        }
    },
    'TRAINING_STEPS': 10000,
    'TEST_STEPS': 100,

    'VISUALIZE_STEPS': 10000,    # Show a visualisatrion every n steps
}
