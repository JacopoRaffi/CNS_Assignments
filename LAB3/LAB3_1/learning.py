import itertools
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TDNNDataset(data.Dataset):
    '''
    Custom Dataset class for sequences (when using a TDNN)

    Atrributes:
    ----------
    x: torch.Tensor
        The input sequence
    y: torch.Tensor
        The target sequence
    window_size: int
        The size of the input window (for tdnn)
    '''
    def __init__(self, x, y, window_size):
        '''
        Initialize the dataset

        Parameters:
        ----------
        x: torch.Tensor
            The input sequence
        y: torch.Tensor
            The target sequence
        window_size: int
            The size of the input window (for tdnn)

        Returns:
        -------
        return: -
        '''
        
        self.x = x
        self.y = y
        self.window_size = window_size

    def __len__(self):
        '''
        Return the length of the dataset

        Returns:
        -------
        return: int
            The length of the dataset
        '''
        return len(self.x)

    def __getitem__(self, idx):
        '''
        Get an item from the dataset (if idx < window_size - 1, pad the input sequence with zeros)

        Parameters:
        ----------
        idx: int
            The index of the item to retrieve

        Returns:
        -------
        return: tuple
            A tuple containing the input sequence and the target value
        '''
        if idx < self.window_size - 1:
            pad_size = self.window_size - 1 - idx
            x = F.pad(self.x, (pad_size, 0)) # apply padding (only for the input sequence)
            return  x[:idx + pad_size + 1], self.y[idx]
        else:
            return self.x[idx - self.window_size + 1:idx + 1], self.y[idx] # no need to apply padding
        

def train_tdnn(model:torch.nn.Module, train_loader, val_loader, lr, weight_decay:float, epochs:int, verbose=True):
    '''
    Train a given model

    Parameters:
    ----------
    model: torch.nn.Module
        The model to train
    train_loader: torch.utils.data.DataLoader
        The training data loader
    val_loader: torch.utils.data.DataLoader
        The validation data loader
    lr: float
        The learning rate
    weight_decay: float
        The weight decay (L2 regularization or Tikhonov Regularization)
    epochs: int
        The number of epochs
    verbose: bool
        Whether to print the training progress

    Returns:
    -------
    return: tuple
        A tuple containing the training and validation loss history
    '''
    loss = torch.nn.MSELoss()
    mae = 0
    optional_loss = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mse_history = [] # store the training loss history
    val_mse_history = [] # store the validation loss history

    for epoch in range(epochs):

        model.train() # set to training mode
        train_mse = 0
        for x, y in train_loader:
            optimizer.zero_grad() # zero the gradients

            out = model(x)
            train_loss = loss(out, y.unsqueeze(1)) # unsqueeze necessary to have the same size for 'out' and 'y' (it doesn't affect the loss)

            train_loss.backward()
            optimizer.step()

            train_mse += train_loss.item()
        
        train_mse /= len(train_loader) # average the loss over the minibatches (number of minbatch given by 'len(train_loader)')
        
        model.eval() # set to evaluation mode
        with torch.no_grad(): # no need to compute gradients (more efficient)
            for x, y in val_loader:
                out = model(x)
                val_mse = loss(out, y.unsqueeze(1)) # unsqueeze necessary to have the same size for 'out' and 'y' (it doesn't affect the loss)
                val_mae = optional_loss(out, y.unsqueeze(1))
                mae = val_mae.item()
        
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse.item())

        if verbose:
            print(f'Epoch {epoch} - Train MSE: {train_mse} - Val MSE: {val_mse.item()}')

    return train_mse_history, val_mse_history, mae

class GridSearch:
    '''
    Implementation of a gridsearch for rnn and tdnn 

    Attributes:
    ----------
    all_config: list
        List of dict containing all possible configurations
    '''
    def __init__(self, hyperparameters:dict):
        '''
        Initialize the gridsearch

        Parameters:
        ----------
        hyperparameters: dict
            Dictionary containing hyperparameters to be tuned

        Returns:
        -------
        return: -
        '''
        self.all_config = [dict(zip(hyperparameters.keys(), config)) for config in itertools.product(*hyperparameters.values())]

    def tdnn_grid_search(self, train_X, train_Y, val_X, val_Y, verbose=False):
        '''
        Perform grid search to find the best hyperparameters for a TDNN

        Parameters:
        ----------
        train_X: torch.Tensor
            Training input data
        train_Y: torch.Tensor
            Training target data
        val_X: torch.Tensor
            Validation input data
        val_Y: torch.Tensor
            Validation target data

        Returns:
        -------
        return: dict
            Return all the configurations with the corresponding training and validation MSE
        '''
        
        model_selection_history = {} # contains all the configurations with the corresponding training and validation MSE (useful for model selection)

        for i, config in enumerate(self.all_config):
            train_dataset = TDNNDataset(train_X, train_Y, window_size=config['window_size'])
            val_dataset = TDNNDataset(val_X, val_Y, window_size=config['window_size'])
            train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
            val_loader = data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
            
            tdnn = TDNN(window_size=config['window_size'], hidden_size=config['hidden_size'], output_size=1).to(device)
            train_h, val_h, mae = train_tdnn(tdnn, train_loader, val_loader, 
                                        lr=config['lr'], weight_decay=config['weight_decay'], epochs=config['epochs'], verbose=False)
            
            model_selection_history[f'config_{i}'] = {**config, 'train_mse': train_h[-1], 'val_mse': val_h[-1], 'val_mae': mae}
            if verbose: 
                print(f'Configuration {i}')
        
        return model_selection_history

    def rnn_grid_search(self):
        pass