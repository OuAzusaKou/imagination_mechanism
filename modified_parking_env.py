
import gym
from highway_env.envs import ParkingEnv


import numpy as np


class ModifiedParkingEnv(ParkingEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,),
                                                dtype=np.float32)  # 修改观测空间为一个包含目标位置信息的数组

    def reset(self):
        # 在重置环境时，首先调用原始的reset方法
        obs = super().reset()

        # 将目标位置信息添加到观测中
        obs = np.concatenate([obs['observation'], obs['desired_goal']])

        return obs

    def step(self, action):
        # 在执行动作后，首先调用原始的step方法
        next_obs, reward, done, info = super().step(action)

        # 将目标位置信息添加到观测中
        next_obs = np.concatenate([next_obs['observation'], next_obs['desired_goal']])

        return next_obs, reward, done, info


# 将修改后的环境注册到Gym中
gym.envs.register(
    id='ModifiedParking-v0',
    entry_point='modified_parking_env:ModifiedParkingEnv',  # 请将your_module_name替换为包含ModifiedParkingEnv的模块名称

)

# 创建环境
# env = gym.make('ModifiedParking-v0')