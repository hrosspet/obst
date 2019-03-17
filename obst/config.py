from obst.env import OneHot1DWorld
from obst.agent import RandomBufferedKerasAgent, WorldModelBufferedKerasAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
        'CONSTRUCTOR': OneHot1DWorld,
        'PARAMS': {
            'size': 1000,
        }
    },
    'AGENT': {
        'CONSTRUCTOR': RandomBufferedKerasAgent,
        # 'CONSTRUCTOR': WorldModelBufferedKerasAgent,
        'PARAMS': {
            'buffer_size': 10000,
            'training_period': 10000,
            'n_actions': 2,
            'input_dim': 1000,
            'batch_size': 32,
            'steps_per_epoch': 1000,
            'epochs': 6,
            'lr': 1e-3,
            'n_layers': 2
        }
    },
    'TRAINING_STEPS': 10000,
    'TEST_STEPS': 100
}
