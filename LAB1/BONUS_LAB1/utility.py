import numpy as np
import itertools
from lsm import LSM

def MAE(Y, Y_hat):
    return np.mean(np.abs(Y - Y_hat))


def grid_search(hyperparameters:dict, train_X:np.ndarray, train_Y:np.ndarray, val_X:np.ndarray, val_Y:np.ndarray):
    all_configs = [dict(zip(hyperparameters.keys(), config)) for config in itertools.product(*hyperparameters.values())]

    best_config = None
    best_MAE = np.inf

    for config in all_configs:
        lsm = LSM(Ne=config['Ne'], Ni=config['Ni'], win_e=config['win_e'], win_i=config['win_i'], w_e=config['w_e'], w_i=config['w_i'])
        lsm.fit(train_X, train_Y)
        
        Y_hat = lsm(val_X)
        mae = MAE(val_Y, Y_hat)
        
        if mae < best_MAE:
            best_MAE = mae
            best_config = config.copy()
    
    return best_config, best_MAE