from obst.env import OneHot1DWorld, OneHot1DCyclicWorld, My2DWorld, Visualizing2DWorld
from obst.models import VectorPreprocessModel, ImagePreprocessModel
from obst.unityenv import ObstTowerWorld
from obst.agent import ExplorationAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
        # 'CONSTRUCTOR': Visualizing2DWorld,
        # 'PARAMS': {
        #     'width': 12, 'height': 12,
        #     'cyclic': False
        # }
        'CONSTRUCTOR': ObstTowerWorld,
        'PARAMS': {
            'path': '/opt/ObstacleTower/obstacletower.x86_64'
        }
    },
    'AGENT': {
        'CONSTRUCTOR': ExplorationAgent,
        'PARAMS': {
            'buffer_size': 10000,
            'training_period': 100,
            'n_actions': 4,
            'repr_size': 16,    # size of internal representation
            'batch_size': 100,
            'steps_per_epoch': 1000,
            'epochs': 2,
            'lr': 1e-3,

            'mode': 'EXPLORE',  # EXPLORE/EXPLOIT

            # 'prep_model': VectorPreprocessModel,
            # 'obs_size': (5,),
            'prep_model': ImagePreprocessModel,
            'obs_size': (168, 168, 3),     # size of input observation
        }
    },
    'TRAINING_STEPS': 100000,
    'TEST_STEPS': 100,

    'VISUALIZE_STEPS': 100,    # Show a visualisatrion every n steps
}
