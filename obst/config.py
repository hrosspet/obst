from obst.env import OneHot1DWorld, OneHot1DCyclicWorld, My2DWorld, Visualizing2DWorld
from obst.agent import ExplorationAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
        'CONSTRUCTOR': Visualizing2DWorld,
        'PARAMS': {
            # 'size': 1000,
            'width': 12, 'height': 12,
            'cyclic': False
        }
    },
    'AGENT': {
        'CONSTRUCTOR': ExplorationAgent,
        'PARAMS': {
            'buffer_size': 10000,
            'training_period': 1000,
            'n_actions': 4,
            'obs_size': 5,     # size of input observation
            'repr_size': 4,    # size of internal representation
            'batch_size': 100,
            'steps_per_epoch': 1000,
            'epochs': 2,
            'lr': 1e-3,

            'mode': 'EXPLORE',  # EXPLORE/EXPLOIT
        }
    },
    'TRAINING_STEPS': 100000,
    'TEST_STEPS': 100,

    'VISUALIZE_STEPS': 100,    # Show a visualisatrion every n steps
}
