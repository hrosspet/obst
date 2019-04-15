from obst.env import OneHot1DWorld, OneHot1DCyclicWorld, My2DWorld, Visualizing2DWorld
from obst.models import VectorPreprocessModel, ImagePreprocessModel, VAEPreprocessModel
from obst.unityenv import ObstTowerWorld
from obst.agent import ExplorationAgent

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': {
    #    'CONSTRUCTOR': Visualizing2DWorld,
    #    'PARAMS': {
    #        'width': 12, 'height': 12,
    #        'cyclic': False
    #    }
        'CONSTRUCTOR': ObstTowerWorld,
        'PARAMS': {
            'path': '/opt/ObstacleTower/obstacletower.x86_64'
        }
    },
    'AGENT': {
        'CONSTRUCTOR': ExplorationAgent,
        'PARAMS': {
            'mode': 'EXPLORE',  # EXPLORE/EXPLOIT

            'training_period': 100,
            'buffer_size': 10000,
            'n_actions': 4,

            'hparams': {    # hyperparameters
                'steps_pe': 1000,
                'epochs': 2,
                'batch_size': 32,

                'lr': 1e-3,
            },

        #    'prep_model': VectorPreprocessModel,
            'prep_model': VAEPreprocessModel,

            'lsizes': {                             # layer sizes
                # 'obs_size': (5,),
                # 'repr_size': 4,
                'obs_size': (168, 168, 3),
                'repr_size': 16,
            },
        }
    },
    'TRAINING_STEPS': 100000,
    'TEST_STEPS': 100,

    'VISUALIZE_STEPS': 100,    # Show a visualisatrion every n steps
}
