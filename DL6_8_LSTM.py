import torch
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

lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(lstm_layer, vocab_size).to(device)

num_epochs, lr, clipping_theta = 160, 2e-2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']


train_and_predict_rnn_pytorch(model, device, corpus_indices, 
                              idx_to_char, char_to_idx, num_epochs, 
                              num_steps, lr, clipping_theta, batch_size, 
                              pred_period, pred_len, prefixes)