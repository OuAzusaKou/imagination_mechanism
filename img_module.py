from jueru.algorithms import SACAlgorithm
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

class MultiHeadSimilarityNetwork(nn.Module):
    def __init__(self, o_dim, a_dim, num_heads, sim_dim):
        super(MultiHeadSimilarityNetwork, self).__init__()
        self.num_heads = num_heads
        self.sim_dim = sim_dim
        # self.fc_o = nn.ModuleList([nn.Linear(o_dim, sim_dim) for _ in range(
        #     num_heads)])
        # self.fc_a = nn.ModuleList([nn.Linear(a_dim, sim_dim) for _ in range(
        #     num_heads)])

        # self.fc_o = nn.Linear(o_dim, sim_dim * num_heads)
        self.fc_o = nn.Sequential(nn.Linear(o_dim, 64), nn.ReLU(), nn.Linear(64, sim_dim * num_heads))
        # self.fc_a = nn.Linear(a_dim, sim_dim * num_heads)
        self.fc_a = nn.Sequential(nn.Linear(a_dim, 64), nn.ReLU(), nn.Linear(64, sim_dim * num_heads))

        # Define layers for processing o and o_n

        # Define layers for processing a and a_n
        # self.fc_a = nn.Linear(a_dim, 64)  # Assuming input_dim is the dimensionality of a and a_n
        # self.fc_a_n = nn.Linear(a_dim, 64)

        # self.o_similarity_matrix = nn.Linear(o_dim,o_dim,bias = False)
        # self.a_similarity_matrix = nn.Linear(a_dim, a_dim, bias=False)

        # Define output layers for each head
        self.heads = nn.Sequential(nn.Linear(num_heads * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        # 128 because we concatenate the similarities from both processing streams

    def forward(self, o, a, o_n, a_n):
        # Process o and o_n



        # Calculate similarities for each head
        # o_emb = data_parallel(self.fc_o, o)
        # o_n_emb = data_parallel(self.fc_o, o_n)
        # a_emb = data_parallel(self.fc_a, a)
        # a_n_emb = data_parallel(self.fc_a, a_n)
        # similarity_o = F.cosine_similarity(o_emb, o_n_emb)
        # similarity_a = F.cosine_similarity(a_emb, a_n_emb)
        # similarities = torch.stack([similarity_o, similarity_a])
        # cat_sim = torch.cat(similarities, dim=0).transpose(1, 0)
        # d_v = self.heads(cat_sim)
        # print('d_v',d_v.shape)

        # o_emb = F.relu(self.fc_o(o))
        # o_n_emb = F.relu(self.fc_o(o_n))
        # a_emb = F.relu(self.fc_a(a))
        # a_n_emb = F.relu(self.fc_a(a_n))

        o_emb = (self.fc_o(o))
        o_n_emb = (self.fc_o(o_n))
        # with torch.no_grad():
        a_emb = (self.fc_a(a))
        a_n_emb =(self.fc_a(a_n))



        similarities = []

        for i in range(self.num_heads):
            # 获取当前块的起始和结束索引
            start_idx = i * self.sim_dim
            end_idx = (i + 1) * self.sim_dim

            # 提取当前块
            o_block1 = o_emb[:,start_idx:end_idx]
            o_block2 = o_n_emb[:,start_idx:end_idx]
            a_block1 = a_emb[:,start_idx:end_idx]
            a_block2 = a_n_emb[:,start_idx:end_idx]
            # print('o_block1', o_block1.shape)
            similarity_o = F.cosine_similarity(o_block1, o_block2)

            similarity_a = F.cosine_similarity(a_block1, a_block2)

            # print('similarity_o',similarity_o.shape)
            # Concatenate the similarities and pass through the output layer for this head
            combined_similarity = torch.stack([similarity_o, similarity_a])
            # print('cs',combined_similarity.shape)
            # diff_value = self.heads[i](combined_similarity)

            similarities.append(combined_similarity)

        # similarities = similarities.stack(similarities)
        # print('s_shape', similarities[0].shape)
        # d_v = torch.sum(similarities,dim=-1)
        cat_sim = torch.cat(similarities, dim=0).transpose(1, 0)

        # print('cat',cat_sim.shape)
        d_v = self.heads(cat_sim)

        return d_v



class SAC_IMG_Algorithm(SACAlgorithm):

    def imagination_update(self, data, num_iters, sim_scale,step):

        def compute_loss_q(data , data_new):
            o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
            # print('i',o.shape)
            # 获取键对应的列表
            # lists_to_shuffle = [data['obs'], data['act'], data['rew'], data['obs2'], data['done']]
            #
            # # 将所有列表合并在一起
            # zipped_lists = list(zip(*lists_to_shuffle))
            #
            # # 打乱合并后的列表
            # random.shuffle(zipped_lists)
            #
            # # 恢复每个键对应的列表
            # shuffled_lists = list(zip(*zipped_lists))
            #
            # # 更新原始数据字典中的键值对
            # new_data = {}
            #
            # new_data['obs'], new_data['act'], new_data['rew'], new_data['obs2'], new_data['done'] = shuffled_lists
            o_n, a_n, r_n, o2_n, d_n = data_new['obs'], data_new['act'], data_new['rew'], data_new['obs2'], data_new['done']

            # o_n = torch.stack(o_n)
            # a_n = torch.stack(a_n)

            # o_n = torch.stack(r_n)
            # o_n = torch.stack(o2_n)
            # o_n = torch.stack(o_n)

            with torch.no_grad():

                q1 = self.agent.functor_dict['critic'].Q1(o, a)
                q2 = self.agent.functor_dict['critic'].Q2(o, a)


            q1_new = self.agent.functor_dict['critic'].Q1(o_n, a_n)
            q2_new = self.agent.functor_dict['critic'].Q2(o_n, a_n)
            q1_backup = q1_new + self.sim_net(o_n, a_n, o, a)
            q2_backup = q2_new + self.sim_net(o_n, a_n, o, a)
            # Bellman backup for Q functions
            # with torch.no_grad():
            #     # Target actions come from *current* policy
            #     a2, logp_a2 = ac.pi(o2)
            #
            #     # Target Q-values
            #     q1_pi_targ = ac_targ.q1(o2, a2)
            #     q2_pi_targ = ac_targ.q2(o2, a2)
            #     q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            #     backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

            # MSE loss against Bellman backup
            loss_q1 = ((q1 - q1_backup) ** 2).mean()
            loss_q2 = ((q2 - q2_backup) ** 2).mean()
            loss_q = loss_q1 + loss_q2

            # Useful info for logging
            # q_info = dict(Q1Vals=q1.detach().numpy(),
            #               Q2Vals=q2.detach().numpy())
            return sim_scale * loss_q, None

        for i in range(num_iters):

            batch_new = self.data_collection_dict['replay_buffer'].sample_batch(self.batch_size)

            data_new = {}
            data_new['obs'] = batch_new['state']
            data_new['act'] = batch_new['action']
            data_new['obs2'] = batch_new['next_state']
            data_new['rew'] = batch_new['reward']
            data_new['done'] = batch_new['done']


            self.agent.optimizer_dict['critic'].zero_grad()
            # q_optimizer.zero_grad()
            loss_q, q_info = compute_loss_q(data, data_new)

            self.sim_optimizer.zero_grad()
            self.agent.optimizer_dict['critic'].zero_grad()

            loss_q.backward()

            self.sim_optimizer.step()
            self.agent.optimizer_dict['critic'].step()
        if num_iters > 0:
            self.writer.add_scalar('sim_loss', loss_q, global_step=step)

        return



    def learn(self, num_train_step, actor_update_freq, imagination_net, sim_lr, sim_scale, reward_scale=1):






        self.sim_net = imagination_net
        self.sim_optimizer = torch.optim.Adam(params=self.sim_net.parameters(),lr=sim_lr)
        self.actor_update_freq = actor_update_freq
        self.agent.functor_dict['actor'].train()
        self.agent.functor_dict['critic'].train()
        step = 0
        episode_num = 0
        average_reward_buf = - 1e6
        while step <= (num_train_step):

            state = self.env.reset()
            episode_reward = 0
            episode_step = 0
            while True:

                if self.render:
                    self.env.render()

                if step < self.start_steps:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.agent.sample_action(state)

                next_state, reward, done, _ = self.env.step(action)

                reward = reward_scale * reward

                episode_step+=1
                if self.max_episode_steps:
                    if episode_step == self.max_episode_steps:
                        done = False

                done_value = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                # print('state',state.shape)
                self.data_collection_dict['replay_buffer'].store(state, action, reward, next_state, done_value)

                state = next_state.copy()

                episode_reward += reward

                if step >= self.min_update_step and step % self.update_step == 0:
                    for _ in range(self.update_step):
                        batch = self.data_collection_dict['replay_buffer'].sample_batch(self.batch_size)

                        self.updator_dict['critic_update'](self.agent, obs=batch['state'], action=batch['action'],
                                                           reward=batch['reward'], next_obs=batch['next_state'],
                                                           not_done=batch['done'], gamma=self.gamma)
                        self.updator_dict['actor_and_alpha_update'](self.agent, obs=batch['state'],
                                                                    target_entropy=-self.agent.functor_dict[
                                                                        'critic'].action_dim)

                        self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'].Q1,
                                                         self.agent.functor_dict['critic'].Q1,
                                                         polyak=self.polyak)
                        self.updator_dict['soft_update'](self.agent.functor_dict['critic_target'].Q2,
                                                         self.agent.functor_dict['critic'].Q2,
                                                         polyak=self.polyak)
                        data_ = {}
                        data_['obs'] = batch['state']
                        data_['act'] = batch['action']
                        data_['obs2'] = batch['next_state']
                        data_['rew'] = batch['reward']
                        data_['done'] = batch['done']
                        # 5
                        self.imagination_update(data_, num_iters=5, sim_scale=sim_scale,step=step)

                step += 1
                # if step >= self.min_update_step and step % self.save_interval == 0:
                #     self.agent.save(address=self.model_address)
                if done or (episode_step == self.max_episode_steps):
                    episode_num += 1
                    self.writer.add_scalar('episode_reward', episode_reward, global_step=step)
                    if self.save_mode == 'eval':
                        if step >= self.min_update_step and episode_num % self.eval_freq == 0:
                            average_reward = self.eval_performance(num_episode=self.eval_num_episode, step=step)
                            if average_reward > average_reward_buf:
                                self.agent.save(address=self.model_address)
                            average_reward_buf = average_reward
                    break