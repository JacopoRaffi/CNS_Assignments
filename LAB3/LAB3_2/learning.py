import itertools
import torch
import statistics as st

from esn import *

def grid_search(hyperparameters:dict, train_x, train_y, val_x, val_y, n_iter:int = 5, verbose:bool = False):
    all_config = [dict(zip(hyperparameters.keys(), config)) for config in itertools.product(*hyperparameters.values())]

    model_selection_history = {}
    mse = torch.nn.MSELoss()

    for i, config in enumerate(all_config):
        input_size = train_x.shape[2]

        train_mse = []
        val_mse = []
        for _ in range(n_iter):
            esn = RegressorESN(input_size=input_size, hidden_size=config['hidden_size'], ridge_regression=config['ridge_regression'],
                            omhega_in=config['omhega_in'], omhega_b=config['omhega_b'], rho=config['rho'], density=1)

            esn.train()        
            h_last = esn.fit(train_x, train_y)
            train_pred = esn(train_x, None)
            train_mse.append(mse(train_pred.unsqueeze(0), train_y.unsqueeze(0)).item()) # unsqueeze necessary for MSELoss torch function to work (it doesn't affect the loss)
            
            esn.eval()
            val_pred = esn(val_x, h_init=h_last)
            val_mse.append(mse(val_pred.unsqueeze(0), val_y.unsqueeze(0)).item())

        model_selection_history[f'config_{i}'] = {**config, 
                                                  'train_mse_mean': st.mean(train_mse), 'train_mse_var': st.variance(train_mse), 
                                                  'val_mse_mean': st.mean(val_mse), 'val_mse_var': st.variance(val_mse)}
        if verbose:
            print(f'Configuration {i}')

    return model_selection_history


