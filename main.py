# from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac

# import tensorflow as tf
import torch
import gym

env_fn = lambda : gym.make('Ant-v3')

ac_kwargs = dict(hidden_sizes=[256,256], activation=torch.nn.ReLU)

logger_kwargs = dict(output_dir='im_sac_ema/to/output_dir', exp_name='experiment_name')

sac(env_fn=env_fn, logger_kwargs=logger_kwargs,ac_kwargs=ac_kwargs,
    steps_per_epoch=4000, epochs=130, replay_size=int(1e6), gamma=0.99,
    polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
    update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
    save_freq=1, sim_scale=0.001, sim_lr=3e-3,crtic_stop=True,ema=True
    )