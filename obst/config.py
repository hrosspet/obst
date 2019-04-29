from obst.env import OneHot1DWorld, OneHot1DCyclicWorld, My2DWorld, Visualizing2DWorld, Twisted2DWorld
from obst.models import VectorPreprocessModel, ImagePreprocessModel
# from obst.unityenv import ObstTowerWorld
from obst.agent import ExplorationAgent

Vizualizing2DWorld_config = {
   'constructor': Visualizing2DWorld,
   'ctor_params': {
       # 'world_def': 'obst/twisted_worlds/twisted_01.txt'
       'world_file': 'obst/twisted_worlds/twisted_02.txt'
   },

   'repr_model': VectorPreprocessModel,     # The model that processes this world's observation data
   'dims': {                                #
       'obs_size': (1,),
       'repr_size': 4,
   }
}

# ObstTowerWorld_config = {
#     'constructor': ObstTowerWorld,
#     'ctor_params': {
#         'path': '/opt/ObstacleTower/obstacletower.x86_64'
#     },
#
#     'repr_model': ImagePreprocessModel,
#     'dims': {
#         'obs_size': (168, 168, 3),
#         'repr_size': 16,
#     }
# }

CONFIG = {
    'TIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    'WORLD': Vizualizing2DWorld_config,
    # 'WORLD': Visualizing2DWorld_config,
    # 'WORLD': ObstTowerWorld_config,
    'AGENT': {
        'constructor': ExplorationAgent,
        'ctor_params': {
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

            'tree_depth': 6,    # Depth of decision tree
        }
    },
    'TRAINING_STEPS': 100000,
    'TEST_STEPS': 100,

    'VISUALIZE_STEPS': 100,    # Show a visualisatrion every n steps
}
