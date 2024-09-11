import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        X = F.one_hot(inputs.long(), self.vocab_size).float()
        Y, self.state = self.rnn(X, state)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, self.state

    def predict(self, prefix, num_chars, device, idx_to_char, char_to_idx, temperature=0.3):
        state = None
        output = [char_to_idx[prefix[0]]]
        
        for t in range(num_chars + len(prefix) - 1):
            X = torch.tensor([output[-1]], device=device).view(-1, 1)
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].to(device), state[1].to(device))
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
        
        return ''.join([idx_to_char[i] for i in output])


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)




# 新的流程图： 定义模型结构→定义预测函数→定义训练函数→使用语料库数据训练模型，优化权重
# → 使用训练好的模型进行预测/文本生成


"""

2. if state is not None: 
序列开始状态可能是None,只有当状态不为None的时候才需要处理
3. if isinstance(state, tuple):
此检查旨在区分RNN类型,某些RNN变体(LSTM,GRU)的状态是一个元组
如LSTM状态包含两个部分:隐藏状态(h)和单元状态(c)
此处代码将两个部分分别移动到指定设备上
"""


"""维度设计 → 时间步数在前、批量大小在中间、输入个数在后
1. 允许处理可变长度序列
2. 便于处理多个并行序列
3. 特征维度放最后是深度学习的常见惯例
"""