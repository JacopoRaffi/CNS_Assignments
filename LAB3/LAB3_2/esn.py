import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reservoir(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, omhega_in:float, omhega_b:float, rho:float, density:float = 1):
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
        super(Reservoir, self).__init__()

        self.input_scaling = nn.Parameter(torch.tensor(omhega_in), requires_grad=False)
        self.rho = nn.Parameter(torch.tensor(rho), requires_grad=False)
        self.hidden_size = nn.Parameter(torch.tensor(hidden_size), requires_grad=False)

        self.W_in = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size, input_size), -omhega_in, omhega_in), requires_grad=False)
        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(hidden_size), -omhega_b, omhega_b), requires_grad=False)

        W_h = nn.init.uniform_(torch.empty(hidden_size, hidden_size), -1, 1)
        W_h = W_h.div_(torch.linalg.eigvals(W_h).abs().max()).mul_(rho).float() # use in-place operations (div_, mul_) to save memory

        self.W_h = nn.Parameter(W_h, requires_grad=False)

    @torch.no_grad()
    def forward(self, input:torch.Tensor, h_init:torch.Tensor, washout:int = 0) -> torch.Tensor:
        '''
        Forward pass through the ESN

        Parameters:
        ----------
        input: torch.Tensor
            Input tensor. Input of Shape (L, input size) or (L, N, input size) if input is batched 
            (L is the length of the sequence, N is the batch size)
        h_init: torch.Tensor
            Initial hidden state (set to zeros if None)
        washout: int
            Number of time steps to ignore

        Returns:
        -------
        return: torch.Tensor
            Output tensor
        '''

        timesteps, batch_size, _ = input.shape
        h = torch.zeros(batch_size, self.hidden_size) if h_init is None else h_init.copy()
        states = []

        for t in range(timesteps):
            h = F.linear(input[t], self.W_in, self.bias) + F.linear(h, self.W_h)
            h = F.tanh(h)
            states.append(h)

        return torch.stack(states[washout:], dim=0) 

            

class ESN(nn.Module):
    pass
            

        

