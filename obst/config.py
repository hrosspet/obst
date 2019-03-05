from obst.env import OneHot1DWorld
from obst.agent import RandomBufferedAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
        'CONSTRUCTOR': OneHot1DWorld,
        'PARAMS': {
            'size': 10,
        }
    },
    'AGENT': {
        'CONSTRUCTOR': RandomBufferedAgent,
        'PARAMS': {
            'buffer_size': 10,
            'training_period': 10,
            'n_actions': 2
        }
    },
    'TRAINING_STEPS': 100,
    'TEST_STEPS': 10
}