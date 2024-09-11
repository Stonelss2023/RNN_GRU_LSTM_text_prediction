import torch
import math
import time
from torch import nn, optim
import sys
sys.path.append("..")
from DL6_3_4jzlyrics_dataset import JaychouDataset
import DL6_3_3jzlyrics_sampler as sampler
from DL6_5_RNN_model import RNNModel, grad_clipping, device

def train_and_predict_rnn_pytorch(model, device, corpus_indices, 
                                  idx_to_char, char_to_idx, num_epochs, 
                                  num_steps, lr, clipping_theta, batch_size, 
                                  pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = sampler.data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
        state = None
        
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = tuple(s.detach() for s in state)
                else:   
                    state = state.detach()

            (output, state) = model(X, state)
            output = output.reshape(-1, len(char_to_idx))
            Y = Y.reshape(-1)

            l = loss(output, Y.long())

            optimizer.zero_grad()
            l.backward()
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * Y.numel()
            n += Y.numel()

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        
        if (epoch + 1) % pred_period == 0:
            print(f'epoch {epoch + 1}, perplexity {perplexity:.2f}, time {time.time() - start:.2f} sec')
            for prefix in prefixes:
                print(' -', model.predict(prefix, pred_len, device, idx_to_char, char_to_idx))

def main():
    jz_dataset = JaychouDataset(num_chars=10000)
    idx_to_char, char_to_idx, vocab_size = jz_dataset.get_vocab_info()
    corpus_indices = jz_dataset.get_corpus_indices()

    num_hiddens = 256
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
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