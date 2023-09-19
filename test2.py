# 使用gym，运行一个倒立摆20次，每次最多100步（随机），中途调整失败即可退出，打印过程步骤和坚持时间

import gym

env = gym.make('CartPole-v0')



for ep in range(20):
    step_num = 0
    o = env.reset()
    ep_reward=0.0
    for i in range(100):
        print('obs', o)
        a = env.action_space.sample()
        step_num += 1
        o, r, d, _ = env.step(a)
        ep_reward+=r
        print('action', a)
        print('reward', r)
        if d:
            print('episode_reward',ep_reward)
            print('step_num',step_num)
            break
