from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt
import numpy as np

from env import World

class ObstTowerWorld(World):
    def __init__(self, path):
        super().__init__()

        # Realtime mode determines whether the environment window will render the scene,
        # as well as whether the environment will run at realtime speed. Set this to `True`
        # to visual the agent behavior as you would in player mode.

        self.env = ObstacleTowerEnv(path, retro=False, realtime_mode=True, worker_id=get_worker_id())

    def step(self, action):
        def compose_act(action):
            # 0. Movement (No-Op/Forward/Back)
            # 1. Camera Rotation (No-Op/Counter-Clockwise/Clockwise)
            # 2. Jump (No-Op/Jump)
            # 3. Movement (No-Op/Right/Left)

            act = np.array([0, 0, 0, 0])

            if action == 0:
                act[0] = 1
            if action == 1:
                act[0] = 2
            if action == 2:
                act[1] = 1
            if action == 3:
                act[1] = 2
            if action == 4:
                act[2] = 1
            if action == 5:
                act[3] = 1
            if action == 6:
                act[3] = 2

            return act

        (obs, reward, done, info) = self.env.step(compose_act(action))

        return obs[0], reward, done, info

    def reset(self):
        self.env.reset()

# The env doesn't close properly on Linux so we have to run it with a different ID each time
# https://github.com/Unity-Technologies/ml-agents/issues/1505#issuecomment-471936096
def get_worker_id(filename=".worker_id"):
    with open(filename, 'a+') as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val
