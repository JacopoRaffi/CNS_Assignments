from torch import nn
import torch

class TDNN(nn.Module):
    def __init__(self, window_size, hidden_size, output_size):
        super(TDNN, self).__init__()

        self.hidden_layer = nn.Linear(window_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def __pad_input(self, x):
        pad_size = self.window_size - x.shape[1]
        
        return torch.cat(torch.zeros(1, pad_size), x)

    def forward(self, x:torch.Tensor):
        if  x.shape[1] < self.window_size:
            x = self.__pad_input(x)

        x = self.hidden_layer(x)
        x = nn.functional.tanh(x)
        x = self.output_layer(x)
        
        return x