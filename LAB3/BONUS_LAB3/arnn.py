import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class AntisymmetricRNN(nn.module):
    def __init__(self, input_size, hidden_size, output_size, euler_step:float, diffusion:float):
        super(AntisymmetricRNN, self).__init__()
        
        self.input_size = nn.Parameter(torch.tensor(input_size), requires_grad=False)
        self.hidden_size = nn.Parameter(torch.tensor(hidden_size), requires_grad=False)
        self.output_size = nn.Parameter(torch.tensor(output_size), requires_grad=False)
        self.euler_step = nn.Parameter(torch.tensor(euler_step), requires_grad=False)
        self.diffusion = nn.Parameter(torch.tensor(diffusion), requires_grad=False)

        sqrt_k = 1 / math.sqrt(hidden_size)
        
        # weight initializitation equal to the initialization of RNN in PyTorch (from Uniform(-sqrt(k), sqrt(k)))
        self.W_in = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, input_size), -sqrt_k, sqrt_k), requires_grad=True)
        self.W_h = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, hidden_size), -sqrt_k, sqrt_k), requires_grad=True)

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size), -sqrt_k, sqrt_k), requires_grad=True)

    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        
        h = torch.zeros(self.hidden_size) if h_init is None else h_init



class TextGeneratorRNN(nn.Module):
    pass