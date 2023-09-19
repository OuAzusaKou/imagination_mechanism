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


device = torch.device('cpu')

env = gym.make('Pendulum-v0')

feature_extractor = FlattenExtractor(env.observation_space)

actor = Sac_actor(
    action_space=env.action_space, hidden_dim=128, feature_extractor=feature_extractor,
    log_std_min=-10, log_std_max=2
)

critic = Sac_critic(
        action_space=env.action_space, feature_extractor=feature_extractor, hidden_dim=128,

    )

log_alpha = torch.tensor(np.log(0.01))
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

sac = SACAlgorithm(agent_class=Sac_agent,
                     functor_dict=functor_dict,
                     lr_dict=lr_dict,
                     updator_dict=updator_dict,
                     data_collection_dict=data_collection_dict,
                     env=env,
                     gamma=0.99,
                     batch_size=100,
                     tensorboard_log="./SAC_tensorboard",
                     render=True,
                     action_noise=0.1,
                     min_update_step=1000,
                     update_step=100,
                     polyak=0.995,
                     save_interval=2000,
                     device=device,
                     max_episode_steps=2000,
                     start_steps=1500,
                     )

sac.learn(num_train_step=50000, actor_update_freq=2)

