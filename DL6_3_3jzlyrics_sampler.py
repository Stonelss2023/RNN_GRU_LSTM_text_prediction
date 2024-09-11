import random
import torch

# 随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

# 相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size #batch_size = 并行序列数 = 单batch样本数
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)#将一维长序列重新组织成多个等长并行序列
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1] #滑动窗口式设计，不跳过任何元素
        yield X, Y

 



"""并行序列处理设计优势
1. 允许并行处理多个(来自不同文本位置的)序列,提高计算效率
2. 保持原始序列中词的顺序,有利于捕捉长距离依赖

妙手：
1. 原始长序列(完整字符级语料库)被分成batch_size个等长并行序列,每个并行序列长度是batch_len
2. 每次迭代从每个并行序列中选取num_steps长度的数据
这样,每个批次包含batch_size个样本,批次的样本数也始终是batch_size"""

"""
1. 维护连续性的优势是相对于随机采样而言的→保持了语料库的原始顺序 
2. 每一个序列中前后时间步的数据之间存在隐藏状态的传递,这种隐藏状态可以携带之前批次的信息并和传入当前批次与之构成关联，
正向传播和反向传播的时候前后信息依赖性均存在 
3. 同一批次中不同并行序列在原始语料库中距离较远,但是作为同一个batch处理的时候可以被捕捉到更广泛的上下文关系"""