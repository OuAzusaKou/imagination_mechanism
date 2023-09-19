# s,a: cuurent state and action
# s_n,a_n: other state and action
# f_q, f_k: feature encoder networks
# loader: minibatch sampler from ReplayBuffer
# m: momentum, e.g. 0.95
# k: head num for Sim
# feature_dim: feature dimension
# q,q_n: shape: [B,k*feature_dim]
f_k.params = f_q.params

for s_n,a_n in loader: # load minibatch from buffer

    q = f_q.forward(s,a)
    q_n = f_k.forward(s_n,a_n)
    q_n = q_n.detach() # stop gradient
    for i in range(k):

        v[i] = Sim(q[i*feature_dim:(i+1)*feature_dim], q_n[i*feature_dim:(i+1)*feature_dim])

    d = T(v)

    loss = MSE(Critic(s,a) + d,Critic(s_n,a_n)
    loss.backward()
    update(f_q.params) # Adam
    f_k.params = m*f_k.params+(1-m)*f_q.params