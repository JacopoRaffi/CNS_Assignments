import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, vocab_size:int):
        super(CharRNN, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        _, h = self.rnn(input, h_init)
        print(h.shape)
        out = self.fc(h)

        return out, h