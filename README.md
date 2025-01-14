# RNN/GRU/LSTM模型在文本预测生成上的性能差异
主要内容：
- 数据预处理
    - 下载与加载 (Jaychou lyrics dataset)
    - 清洗与标准化
    - 采样
    - 数据集呈现
- RNN模型结构
    - 
- GRU
- LSTM
- 训练过程

# 1. Preprocessing


```python
# Establish the lexicon
def preprocess(corpus_chars, num_chars=10000):
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[:num_chars]

    idx_to_char = list(set(corpus_chars))
    char_to_idx = {char: i for i, char in enumerate(idx_to_char)}
    vocab_size = len(cahr_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]

    return corpus_chars, idx_to_char, char_to_idx, vocab_size, corpus_indices

# Sampling
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' is torch.cuda.is_available() else 'cpu')
    corpus_indices = trorch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i+1, i + num_stpes + 1]
        yield X, Y
```
【并行序列处理的设计优势】
1. 允许并行处理多个序列 → 提高效率
2. 保持原始序列中词的顺序，有利于捕捉长距离依赖 (和随机采样相比)

【妙手】
1. 原始长序列(完整字符级语料库)被分成batch_size个等长并行序列,每个并行序列长度时batch_len
2. 每次迭代从每个并行序列中选取num_steps长度的数据
如此，每个批次的样本数始终保持为 batch_size
batch_size = 并行序列数 = 单batch样本数
# 2. Dataset Synthesized


```python
class JaychouDataset:
    def __init__(self, num_chars=10000):
        self.num_chars = num_chars
        self.corpus_chars = None
        self.idx_to_char = None
        self.char_to_idx = None
        self.vocab_size = None
        self.corpus_indices = None

    def load_data(self):
        if self.corpus_chars is None:
            print("Loading data...")
            self.corpus_chars = download_data()
            (self.corpus_chars, 
             self.idx_to_char,
             self.vocab_size,
             self.vocab_size, 
             self.corpus_indices
            ) = preprocess_data(self.corpus_chars, self.num_chars)
            print("Data loaded and processed.")

    def get_random_iter(self, batch_size, num_steps):
        self.load_data()
        return data_iter_random(self.corpus_indices, batch_size, num_steps)

    def get_consecutive_iter(self, batch_size, num_steps):
        self.load_data()
        return data_iter_consecutive(self.corpus_indices, batch_size, num_steps)

    def get_corpus_chars(self):
        self.load_data()
        return self.corpus_chars

    def get_vocab_info(self):
        self.load_data()
        return self.idx_to_char, self.char_to_idx, self.vocab_size

    def get_corpus_indices(self):
        self.load_data()
        return self.corpus_indices
```
封装完整的数据处理流程，统一管理数据集相关信息. 所谓"接口",就是类/模块对外提供的调用方法
1. 它们封装了内部复杂的数据处理逻辑，但对外提供简单统一的调用方式。这个项目写成封装形式的根本原因在于：sampling脚本定义了两种采样方式。
2. 封装成类的第二个优势：延迟加载(Lazy Loading) → 在真正需要使用数据时才进行加载器，而不是在创建对象时就立即加载所有数据。在__init__过程中只设置参数，不加载数据。
1. hidden_size != hidden_layer → hidden_size是每个时间步隐藏状态的向量维度大小，也等价于抽象意义上的神经元数量。 其数量大小所代表的具体含义不是预定义的、不是可解释的→学出来的抽象特征
较大的hidden_size可以捕捉更复杂的模式，但会增加过拟合风险
2. F.one_hot期望参数→输入索引+词汇表大小
输入索引是DL6_3_2jzlyrics_processor中的corpus_indices
.long()把输入转换为长整型，因为one_hot函数期望整数索引
.float()将整数类型转换为浮点类型→神经网络通常使用浮点数计算→支持梯度计算和反向传播
3. Y.shape[-1]表示Y张量的最后一维大小（对应隐藏状态大小）
重塑后：dim1→时间步*批次 dim2→hidden_size  →  方便应用全连接层
output形状 → [sequence_length * batch_size, vocab_size] → 包含所有时间步预测结果的
4. dense操作每个时间步输出的隐藏状态→接受隐藏状态的输出向量维度，映射到vocab上，预测下一步的词汇概率分布
5. rnn_layer.bidirectional 处理双向RNN的hidden_size问题→判断是否要乘2
6. return output, self.state → output 是模型主要输出，针对每一步的字符概率分布预测[batch_size*sequence_length, vocab_size]
   self.state 对于RNN→最终隐藏状态
              对于LSTM→(hidden_state, cell_state)
              对于GRU→最后一个时间步隐藏状态
   [num_layers*num_directions, batch_size, hiddden_size]

7. tensor_size的关键点解释
(1) inputs→[batch_size, sequence_length] 例如[64, 100]表示64个样本，样本时间步为100
(2) one_hot编码以后→[batch_size, sequence_length, vocab_size] → [64, 100, 5000]
(3) RNN层 Y→[batch_size, sequence_length, hidden_size] → [64, 100, 256](信息抽象提炼)
          state:
           RNN/GRU → [num_layers*num_directions, batch_size, hidden_size]
           LSTM → [hidden_state, cell_state] → [2, 64, 256] （假设有2层单向RNN）
8. 在forward方法中又传入一个state→it can be self.state defined in the class RNNModel, it can also be a brand new state conveyed by the user. This flexibility entitles the user with the right to choose the initial state when resetting.
   

# 3. RNN_Model


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None #初始隐藏状态为空

    def forward(self, inputs, state):
        X = F.one_hot(inputs.long(), self.vocab_size).float()
        Y, self.state = self.rnn(X, state)
        output = self.dense (Y.reshape(-1, Y.shape[-1]))
        return output, self.state

    def predict(self, prefix, num_chars, device, idx_to_char, char_to_idx, temperature=0.3):
        state = None
        output = [char_to_idx[prefix[0]]] #使用给定的前缀(prefix)初始化生成过程

        for t in range(num_chars + len(prefix) - 1):
              X = torch.tensor([output[-1]], device=device).view(-1, 1)
              if state is not None:
                  if isinstance(statte, tuple):
                    state = state[0].to(device), state[1].to(device)) #LSTM的隐藏状态→(h, c)
                  else:
                      state = state.to(device)

              (Y, state) = self(X, state)

              if t < len(prefix) - 1:
                  output.append(char_to_idx[prefix[t + 1]])
              else:
                  scaled_logits = Y / temperature
                  probs = F.softmax(scaled_logits, dim=1)
                  next_char_idx = torch.multinomial(probs, num_samples=1).item()
                  output.append(next_char_idx)

        return ''. join([idx_to_char[i] for i in output]) #把字符索引映射回字符,拼接成字符串返回


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grd.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

```

2. X = torch.tensor([output[-1]], device=device).view(-1, 1)
准备输入数据 → 获取output列表最后一个元素（最新预测字符的索引）重塑张量为一个列向量
3. if isinstance(state, tuple):检查状体是否是元组 → 隐藏状态+单元状态

4. softmax运算指定 dim=1 → Y的通常形状是(batch_size, vocab_size),在词汇表维度应用softmax确保所有字符的可能概率和为1
5. torch.multinomial(probs, num_samples=1) 从给定的probs概率分布中采样；num_samples=1 → 只采样一次

# 4. RNN_Train            


```python
import torch
import math
import time
from torch import nn, optim
import sys
sys.path.append("..")
from DL_6_3_4jzlyrics_dataset import JaychouDataset
from DL6_5_RNN_model import RNNModel, grad_clipping, device

def train_and_predict_rnn_pytorch(model, device, corpus_indices, idx_to_char, char_to_idx, 
                                  num_epochs,num_steps, clipping_theta, batch_size,
                                  lr, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modal.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    for epoch in range(num_epochs):
        for epoch in range(num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            data_iter = sampler.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
            state = None

            for X, Y in date_iter:
                if state is not None:
                    if isinstance(state, tuple):
                        state = tuple(s.detach() for s in state) 
                    else:
                        state = state.detach() #清除隐藏状态的梯度信息
            (output, state) = model(X, state)
            output = output.reshape(-1, len(char_to_idx)) #(batch_size*num_steps, vocab_size)
            Y = Y.reshape(-1) #(batch_size*num_steps)

            l = loss(output, Y.long())

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * Y.numel() #损失标量化 乘以批量样本数
            n += Y.numel() #更新样本总数

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf') #指数运算溢出则设置perplexity为无穷

        if (epoch + 1) % pred_period == 0: 
            print(f'epoch {epoch + 1}, perplexity {perplexity:.2f}, time {time.time() - start:.2f} sec')
            for prefix in prefixes:
                print(' -', model.predict(prefix, pred_len, device, # pred_len指定生成文本的长度
                                          idx_to_char, char_to_idx)) #使用前缀调用模型的predict方法生成文本
            
def main():
    jz_dataset = JaychouDataset(num_chars=10000)
    idx_to_char, char_to_idx, vocab_size = jz_dataset.get_vocab_info()
    corpus_indices = jz_dataset.get_corpus.indices()

    num_hiddens = 256
    rnn_layer = nn.RNN(inputs_size=vocab_size, hidden_size=num_hiddens)
    model = RNNModel(rnn_layer, vocab_size).to(device)

    num_epochs, batch_size, lr, clipping_theta = 200, 32, 1e-3, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    num_steps = 35

    train_and_predict_rnn_pytorch(model, device, corpus_indices,
                                  idx_to_char, char_to_idx, num_epochs, 
                                  num_steps, lr, clipping_theta, batch_size, 
                                  pred_period, pred_len, prefixes)

if __name__ == "__main__":
    main()
            
            

```
相较于传统的训练流程,多出的部分主要包括：
    '隐藏状态梯度信息处理'
    'RNN Varients的多个隐藏状态处理'
    '依据连续采样和封装的数据维度变化 → loss计算之前'
    'optimization前的梯度裁剪步骤'
    '评判标准计算:perplexity'
# 5. GRU_model


```python
num_hiddens = 256
num_steps = 35
batch_size = 16

rnn_layer = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
model = RNNModel(rnn_layer, vocab_size).to(device)

num_epochs, lr, clipping_theta = 150, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

train_and_predict_rnn_pytorch(model, device, corpus_indices, 
                              idx_to_char, char_to_idx, num_epochs, 
                              num_steps, lr, clipping_theta, batch_size, 
                              pred_period, pred_len, prefixes)
```

# 6. LSTM_model 


```python
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
```
