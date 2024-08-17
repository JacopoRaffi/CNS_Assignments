import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class AntisymmetricRNN(nn.module): 
    def __init__(self, input_size, hidden_size, euler_step:float, diffusion:float):
        super(AntisymmetricRNN, self).__init__()
        
        self.input_size = nn.Parameter(torch.tensor(input_size), requires_grad=False)
        self.hidden_size = nn.Parameter(torch.tensor(hidden_size), requires_grad=False)
        self.euler_step = nn.Parameter(torch.tensor(euler_step), requires_grad=False)
        self.diffusion = nn.Parameter(torch.tensor(diffusion), requires_grad=False)

        sqrt_k = 1 / math.sqrt(hidden_size)
        
        # weight initializitation equal to the initialization of RNN in PyTorch (from Uniform(-sqrt(k), sqrt(k)))
        self.W_in = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, input_size), -sqrt_k, sqrt_k), requires_grad=True)
        self.W_h = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, hidden_size), -sqrt_k, sqrt_k), requires_grad=True)

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size), -sqrt_k, sqrt_k), requires_grad=True)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        h = torch.zeros(self.hidden_size) if h_init is None else h_init
        diffusion_eye = torch.eye(self.hidden_size) * self.diffusion

        for x in input:
            h = h + self.euler_step * (F.linear(h, (self.W_h - self.W_h.T - diffusion_eye)) + F.linear(x, self.W_in, self.bias))

        return h # return only the last hidden state

class CharRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_dim, euler_step:float, diffusion:float):
        super(CharRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = AntisymmetricRNN(embedding_dim, hidden_size, euler_step, diffusion)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor, temperature:float=1.0):
        x = self.embedding(input)

        h = self.rnn(x, h_init)
        out = self.fc(h)
        out = torch.div(out, temperature)

        return F.softmax(out, dim=1)