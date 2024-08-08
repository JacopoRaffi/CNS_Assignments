import numpy as np
import itertools
from lsm import LSM

def MAE(Y, Y_hat):
    '''
    Compute Mean Absolute Error

    Parameters:
    ----------
    Y: np.ndarray
        Target values
    Y_hat: np.ndarray
        Predicted values

    Returns:
    --------
    return: float
        Return the Mean Absolute Error
    '''
    return np.mean(np.abs(Y - Y_hat))


def grid_search(hyperparameters:dict, train_X:np.ndarray, train_Y:np.ndarray, val_X:np.ndarray, val_Y:np.ndarray):
    '''
    Perform grid search to find the best hyperparameters

    Parameters:
    ----------
    hyperparameters: dict
        Dictionary containing hyperparameters to be tuned
    train_X: np.ndarray
        Training input data
    train_Y: np.ndarray
        Training target
    val_X: np.ndarray
        Validation input data
    val_Y: np.ndarray
        Validation target

    Returns:
    --------
    return: tuple
        Return the best hyperparameters, best validation MAE and the corresponding training MAE
    '''

    # create a list of dict contaning all possible configurations
    all_configs = [dict(zip(hyperparameters.keys(), config)) for config in itertools.product(*hyperparameters.values())]

    best_config = None
    best_MAE = np.inf

    for config in all_configs:
        lsm = LSM(Ne=config['Ne'], Ni=config['Ni'], win_e=config['win_e'], win_i=config['win_i'], w_e=config['w_e'], w_i=config['w_i'])
        lsm.fit(train_X, train_Y)
        
        Y_hat = lsm(val_X)
        mae = MAE(val_Y, Y_hat)
        
        if mae < best_MAE: # update best configuration
            best_MAE = mae
            best_config = config.copy()
            y_train_hat = lsm(train_X)
            training_MAE = MAE(train_Y, y_train_hat)
    
    return best_config, best_MAE, training_MAE