import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    '''
    CharRNN model

    Attributes: 
    ------------
    rnn: nn.GRU
        GRU layer
    fc: nn.Linear
        Fully connected layer
    '''
    def __init__(self, input_size:int, hidden_size:int, vocab_size:int):
        '''
        Parameters:
        ------------
        input_size: int
            Size of the input
        hidden_size: int
            Size of the hidden layer
        vocab_size: int
            Size of the vocabulary
        '''
        super(CharRNN, self).__init__()
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        '''
        Compute the output

        Parameters:
        ------------
        input: torch.Tensor
            the input, shape is (L, N D)
        h_init: torch.Tensor
            the initial state
        
        Returns:
        --------
        return: tuple
            the output and the last hidden state computed by the GRU Layer
        '''
        _, h = self.rnn(input, h_init)
        out = self.fc(h.squeeze(0))

        return out, h
    
class OneHotDataset:
    def __init__(self, seq_len, chr_step, text, char_to_idx, one_hot_size, device):
        '''
        Create a datasate of sequences encoded in one-hot style

        Parameters:
        -----------
        seq_len: int
            desired length of a single sequence
        chr_stp: int
            how many characters to skip for every sequence
        text: str
            input text
        char_to_idx: dict
            a dict mapping char to index
        one_hot_size: int
            size of the one-hot vectors
        device: str
            device should be "cuda" or "cpu"

        '''
        sequences = []
        next_chars = []

        for i in range(0, len(text) - seq_len, chr_step):
            sequences.append(text[i:i+seq_len]) # input sequence
            next_chars.append(text[i+seq_len]) # char to predict
        
        self.x = torch.zeros(len(sequences), seq_len, one_hot_size) # shape (N, L ,D)
        self.y = torch.zeros(len(sequences), one_hot_size) # shape (N, D)

        # the idea is to encode every char of a sequence in one-hot vector and this sequence is the input while the target is the next char
        for i, sentence in enumerate(sequences):
            for t, char in enumerate(sentence):
                self.x[i, t, char_to_idx[char]] = 1
            self.y[i, char_to_idx[next_chars[i]]] = 1

        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __getitem__(self, idx):
        '''
        Retrieve an element of the dataset

        Parameters:
        -----------
        idx: int
            index to retrieve
        
        Returns:
        --------
        return: tuple
            the input sequence and the corresponding char target
        '''
        return self.x[idx, :, :], self.y[idx]

    def __len__(self):
        '''
        Simply compute the len of the dataset

        Returns:
        --------
        return: torch.Tensor
            the length of the whole dataset
        '''
        return self.y.shape[0]