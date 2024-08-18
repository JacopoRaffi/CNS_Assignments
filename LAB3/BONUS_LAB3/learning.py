import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, vocab_size:int):
        super(CharRNN, self).__init__()
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        _, h = self.rnn(input, h_init)
        out = self.fc(h.squeeze(0))

        return out, h
    
class OneHotDataset:
    def __init__(self, seq_len, chr_step, text, char_to_idx, one_hot_size, device):
        sequences = []
        next_chars = []

        for i in range(0, len(text) - seq_len, chr_step):
            sequences.append(text[i:i+seq_len]) # input sequence
            next_chars.append(text[i+seq_len]) # char to predict
        
        self.x = torch.zeros(len(sequences), seq_len, one_hot_size) # shape (L, N ,D)
        self.y = torch.zeros(len(sequences), one_hot_size)

        for i, sentence in enumerate(sequences):
            for t, char in enumerate(sentence):
                self.x[i, t, char_to_idx[char]] = 1
            self.y[i, char_to_idx[next_chars[i]]] = 1

        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx]

    def __len__(self):
        return self.y.shape[0]