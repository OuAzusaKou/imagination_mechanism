import random
import torch

# 假设这是你的数据，每个键对应一个多维张量
data = {
    'obs': torch.tensor([[1, 2], [3, 4], [5, 6]]),
    'act': torch.tensor([[0, 1], [2, 3], [4, 5]]),
    'rew': torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    'obs2': torch.tensor([[6, 5], [4, 3], [2, 1]]),
    'done': torch.tensor([[1, 0], [0, 1], [1, 0]])
}

# 获取键对应的多维张量
tensors_to_shuffle = [data['obs'], data['act'], data['rew'], data['obs2'], data['done']]

# 将所有多维张量合并在一起
zipped_tensors = list(zip(*tensors_to_shuffle))

# 打乱合并后的多维张量
random.shuffle(zipped_tensors)

# 恢复每个键对应的多维张量
shuffled_tensors = list(zip(*zipped_tensors))

# 更新原始数据字典中的键值对
data['obs'], data['act'], data['rew'], data['obs2'], data['done'] = shuffled_tensors

# 打印打乱后的数据
print(data)