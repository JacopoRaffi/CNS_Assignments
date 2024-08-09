import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

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
        # load entire dataset in memory (WARNING: only because the dataset is small)
        self.x = x.to(device)
        self.y = y.to(device)
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
        

def train_model(model, train_loader, val_loader, lr, weight_decay, epochs, verbose=True):
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
    mae_loss = torch.nn.L1Loss() # mean absolute error loss (WARNING: it is only used for evaluation...I personally visualize it better)
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
                val_mae = mae_loss(out, y.unsqueeze(1))
        
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse.item())

        if verbose:
            print(f'Epoch {epoch} - Train MSE: {train_mse} - Val MSE: {val_mse.item()} - Val MAE: {val_mae.item()}')

    return train_mse_history, val_mse_history

