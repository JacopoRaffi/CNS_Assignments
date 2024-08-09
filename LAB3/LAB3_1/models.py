from torch import nn
import torch.nn.functional as F
import torch

class TDNN(nn.Module):
    '''
    Implementation of a TDNN model (multilayer perceptron with tanh activation function for the hidden layer)

    Attributes:
    ----------
    window_size: int
        The size of the input window
    hidden_layer: nn.Linear
        The hidden layer of the model (activation function is tanh)
    output_layer: nn.Linear
        The output layer of the model (no activation function)
    '''
    def __init__(self, window_size:int, hidden_size:int, output_size:int = 1):
        '''
        Initialize the TDNN model

        Parameters:
        ----------
        window_size: int
            The size of the input window
        hidden_size: int
            The size of the hidden layer
        output_size: int
            The size of the output layer
        '''
        super(TDNN, self).__init__()

        self.window_size = window_size
        self.hidden_layer = nn.Linear(window_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor):
        '''
        Forward pass of the model

        Parameters:
        ----------
        x: torch.Tensor
            The input tensor of shape (batch_size, window_size)
        
        Returns:
        -------
        x: torch.Tensor
            The output tensor of shape (batch_size, output_size)
        '''
        x = self.hidden_layer(x)
        x = F.relu(x) 
        x = self.output_layer(x)
        
        return x
    

class VanillaRNN(nn.Module):
    '''
    Implementation of a vanilla RNN (for regression task)

    Attributes:
    ----------
    rnn: torch.nn.Module
        recurrent layer of the vanilla model
    readout: torch.nn.Module
        output layer of the model
    '''
    def __init__(self, input_size:int, hidden_size:int, output_size:int = 1):
        '''
        Initialize the VanillaRNN model

        Parameters:
        ----------
        input_size: int
            The size of the input
        hidden_size: int
            The number of features in the hidden state h
        output_size: int
            The size of the output layer (readout) 

        Returns:
        -------
        return: -
        '''
        super(VanillaRNN, self).__init__()

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        '''
        Forward pass of the model

        Parameters:
        ----------
        x: torch.Tensor
            The input sequence
        h0: torch.Tensor
            The initial hidden state of the recurrent layer

        Returns:
        -------
        out: tuple
            the predicted output of the model and the last hidden state of the recurrent layer
        '''
        out, hn = self.rnn(x, h0)

        return self.readout(out), hn
