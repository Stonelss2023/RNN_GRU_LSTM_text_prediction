"""裁剪梯度无法解决梯度衰减的问题
因此，循环神经网络在实际中较难捕捉时间序列中 时间步距离较大的依赖关系

gated recurrent neural network → 门控循环神经网络
gated recurrent unit → GRU 通过学习的门来控制信息流动"""

import torch
import math
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import time

import sys
sys.path.append("..")
from DL6_3_4jzlyrics_dataset import JaychouDataset
import DL6_3_3jzlyrics_sampler as sampler
from DL6_5_RNN_model import RNNModel, device
from DL6_5_RNN_train import train_and_predict_rnn_pytorch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {device}")


jz_dataset = JaychouDataset(num_chars=10000)

print("Loading vocabulary information...")
idx_to_char, char_to_idx, vocab_size = jz_dataset.get_vocab_info()

print("Loading corpus indices...")
corpus_indices = jz_dataset.get_corpus_indices()

# 模型参数
num_hiddens = 256
num_steps = 35
batch_size = 16

# 创建GRU层和模型
rnn_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(rnn_layer, vocab_size).to(device)

num_epochs, lr, clipping_theta = 150, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']


train_and_predict_rnn_pytorch(model, device, corpus_indices, 
                              idx_to_char, char_to_idx, num_epochs, 
                              num_steps, lr, clipping_theta, batch_size, 
                              pred_period, pred_len, prefixes)