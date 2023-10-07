import gym
import numpy as np
import torch

from jueru.Agent_set import Agent, Sac_agent
from jueru.algorithms import BaseAlgorithm, SACAlgorithm
from jueru.datacollection import Replay_buffer
from jueru.updator import critic_updator_ddpg, actor_updator_ddpg, soft_update, actor_and_alpha_updator_sac, \
    critic_updator_sac
from jueru.user.custom_actor_critic import MLPfeature_extractor, ddpg_actor, ddpg_critic, FlattenExtractor, Sac_actor, \
    Sac_critic

import highway_env
import modified_parking_env
from img_module import MultiHeadSimilarityNetwork, SAC_IMG_Algorithm

seed = 13
np.random.seed(seed)
torch.manual_seed(seed)

# 如果你使用GPU，还可以设置GPU随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device('cuda:0')
# env = gym.make('Pendulum-v0')

# env = gym.make('Humanoid-v3')
env = gym.make('Ant-v3')
# env = gym.make('HalfCheetah-v3')

# env = gym.make("ModifiedParking-v0")
# env.configure({
#     "observation": {
#         "type": "KinematicsGoal",
#         "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
#         "scales": [100, 100, 5, 5, 1, 1],
#         "normalize": False
#     },
#     "action": {
#         "type": "ContinuousAction"
#     },
#     "simulation_frequency": 15,
#     "policy_frequency": 5,
#     "screen_width": 600,
#     "screen_height": 300,
#     "centering_position": [0.5, 0.5],
#     "scaling": 7,
#     "show_trajectories": False,
#     "render_agent": True,
#     "offscreen_rendering": False
# })
print(env.observation_space)
print(env.action_space)
feature_extractor = FlattenExtractor(env.observation_space)

actor = Sac_actor(
    action_space=env.action_space, hidden_dim=128, feature_extractor=feature_extractor,
    log_std_min=-10, log_std_max=2
).to(device)

critic = Sac_critic(
        action_space=env.action_space, feature_extractor=feature_extractor, hidden_dim=128,

    ).to(device)

log_alpha = torch.tensor(np.log(0.01)).to(device)
log_alpha.requires_grad = True

data_collection_dict = {}

data_collection_dict['replay_buffer'] = Replay_buffer(env=env, size=1e6,device=device)

functor_dict = {}

lr_dict = {}

updator_dict = {}

functor_dict['actor'] = actor

functor_dict['critic'] = critic

functor_dict['critic_target'] = None

functor_dict['log_alpha'] = log_alpha

lr_dict['actor'] = 1e-3

lr_dict['critic'] = 1e-3

lr_dict['critic_target'] = 1e-3

lr_dict['log_alpha'] = 1e-3

updator_dict['actor_and_alpha_update'] = actor_and_alpha_updator_sac

updator_dict['critic_update'] = critic_updator_sac

updator_dict['soft_update'] = soft_update

img_net = MultiHeadSimilarityNetwork(o_dim=env.observation_space.shape[0],a_dim=env.action_space.shape[0], num_heads=10,sim_dim=16).to(device)

sac = SAC_IMG_Algorithm(agent_class=Sac_agent,
                     functor_dict=functor_dict,
                     lr_dict=lr_dict,
                     updator_dict=updator_dict,
                     data_collection_dict=data_collection_dict,
                     env=env,
                     gamma=0.99,
                     batch_size=100,
                     tensorboard_log="./SAC_tensorboard",
                     render=False,
                     action_noise=0.1,
                     min_update_step=1000,
                     update_step=100,
                     polyak=0.995,
                     save_interval=2000,
                     device=device,
                     start_steps=1500,
                     max_episode_steps=1000,
                     save_mode='eval',
                     eval_freq=10,
                     )
# 1e-3 0.001
sac.learn(num_train_step=500000, actor_update_freq=2,imagination_net=img_net,sim_lr=1e-2, sim_scale=0.002, reward_scale=2.0)
