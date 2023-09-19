# from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac

# import tensorflow as tf
import torch
import gym

env_fn = lambda : gym.make('Pendulum-v0')

# ac_kwargs = dict(hidden_sizes=[64,64], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

sac(env_fn=env_fn, logger_kwargs=logger_kwargs,
    steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99,
    polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=1000,
    update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
    save_freq=1
    )