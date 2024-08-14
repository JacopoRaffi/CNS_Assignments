import numpy as np
from sklearn.linear_model import Ridge
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
    def forward(self, input:torch.Tensor, h_init:torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the ESN

        Parameters:
        ----------
        input: torch.Tensor
            Input tensor. Input of Shape (L, N, input size)
            (L is the length of the sequence, N is the batch size)
        h_init: torch.Tensor
            Initial hidden state (set to zeros if None)

        Returns:
        -------
        return: torch.Tensor
            Output tensor
        '''

        timesteps, batch_size, _ = input.shape
        h = torch.zeros(batch_size, self.hidden_size) if h_init is None else h_init
        states = []

        for t in range(timesteps):
            h = F.linear(input[t], self.W_in, self.bias) + F.linear(h, self.W_h)
            h = F.tanh(h)
            states.append(h)

        return torch.stack(states, dim=0) 

            

class RegressorESN(nn.Module):
    '''
    Echo State Network (ESN) with linear readout for regression task

    Attributes:
    ----------
    reservoir: Reservoir
        Reservoir layer of the ESN
    readout: Ridge
        Linear ridge regression of scikit-learn
    states: torch.Tensor
        States of the reservoir layer (used to avoid recomputing states during training)
    '''

    def __init__(self, input_size:int, hidden_size:int, ridge_regression:float, 
                 omhega_in:float, omhega_b:float, rho:float, density:float = 1):
        '''
        Initialize ESN with the given parameters

        Parameters:
        ----------
        input_size: int
            Size of the input
        hidden_size: int
            Size of the hidden layer
        ridge_regression: float
            Regularization parameter for ridge regression
        omhega_in: float
            Input scaling for Reservoir layer
        omhega_b: float
            Bias scaling for Reservoir layer
        rho: float
            Desired Spectral radius of the hidden reservoir layer weight matrix
        '''
        
        super(RegressorESN, self).__init__()
        
        self.reservoir = Reservoir(input_size, hidden_size, omhega_in, omhega_b, rho, density)
        self.readout = Ridge(alpha=ridge_regression) # linear ridge regression of scikit-learn
        self.states = None
    
    def fit(self, input:torch.Tensor, target:torch.Tensor, washout:int = 0):
        '''
        Fit the ESN to the given input and target

        Parameters:
        ----------
        input: torch.Tensor
            Input tensor. Input of Shape (L, N, input size)
            (L is the length of the sequence, N is the batch size)
        target: torch.Tensor
            Target tensor
        washout: int
            Number of time steps to ignore

        Returns:
        -------
        return: torch.Tensor
            Last state of the Reservoir layer
        '''
        self.states = self.reservoir(input, h_init=None).squeeze(1)
        self.readout.fit(self.states[washout:, :], target[washout:])

        return self.states[-1] # return last state

    @torch.no_grad()
    def forward(self, input:torch.Tensor, h_init:torch.Tensor):
        '''
        Forward pass through the ESN

        Parameters:
        ----------
        input: torch.Tensor
            Input tensor. Input of Shape (L, N, input size)
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
        if self.training: # avoid to recompute states
            states = self.states # states already computed during fitting
        else:
            states = self.reservoir(input, h_init=h_init).squeeze(1)

        return torch.from_numpy(self.readout.predict(states))
            

        

