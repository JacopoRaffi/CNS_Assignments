import numpy as np
import torch
import torch.nn as nn

class ESN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, omhega_in:float, omhega_b:float, rho:float):
        '''
        Initialize Echo State Network (ESN) with the given parameters

        Parameters:
        ----------
        input_size: int
            Size of the input
        hidden_size: int
            Size of the hidden layer
        omhega_in: float
            Input scaling
        omhega_b: float
            Bias scaling
        rho: float
            Desired Spectral radius of the hidden recurrent layer weight matrix

        Returns:
        -------
        return: -
        '''
        super(ESN, self).__init__()

        self.input_scaling = nn.Parameter(torch.tensor(omhega_in), requires_grad=False)
        self.rho = nn.Parameter(torch.tensor(rho), requires_grad=False)
        self.hidden_size = nn.Parameter(torch.tensor(hidden_size), requires_grad=False)

        self.W_in = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, input_size), -omhega_in, omhega_in), requires_grad=False)
        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, 1), -omhega_b, omhega_b), requires_grad=False)

        W_h = nn.init.uniform_(torch.empty(hidden_size, hidden_size), -1, 1)
        W_h = W_h.div_(torch.linalg.eigvals(W_h).abs().max()).mul_(rho).float() # use in-place operations to save memory

        self.W = nn.Parameter(W_h, requires_grad=False)

        

