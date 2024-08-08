from torch import nn
import torch.nn.functional as F
import torch

class TDNN(nn.Module):
    '''
    Implementation of a TDNN model (multilayer perceptron with tanh activation function for the hidden layer)

    Attributes:
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

        # Pad the input if it is smaller than the window size
        if  x.shape[1] < self.window_size:
            pad_size = self.window_size - x.shape[1]
            x = F.pad(x, (0, pad_size))

        x = self.hidden_layer(x)
        x = F.tanh(x)
        x = self.output_layer(x)
        
        return x