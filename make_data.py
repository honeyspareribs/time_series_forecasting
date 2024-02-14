import numpy as np
import math
import os

# 如果没有data目录，则创建
if not os.path.exists('data/train'):
    os.makedirs('data/train')
if not os.path.exists('data/test'):
    os.makedirs('data/test')

# 定义数据长度
length = 1000

# 定义不同的相位
# phases = np.random.rand(1000) * 2 * math.pi
phases = np.random.rand(10) * 2 * math.pi

# 对每个相位生成cos序列
for i, phase in enumerate(phases):
    sequence = [math.cos(2 * math.pi * j / length + phase) for j in range(length)]
    # 数据保存为numpy数组
    sequence = np.array(sequence,dtype=np.float32)
    # 保存为.npy文件
    #np.save('./data/train/cos_sequence_{}.npy'.format(i), sequence)
    np.save('./data/test/cos_sequence_{}.npy'.format(i), sequence)

