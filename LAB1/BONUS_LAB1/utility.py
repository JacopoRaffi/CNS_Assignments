import numpy as np

def MAE(Y, Y_hat):
    return np.mean(np.abs(Y - Y_hat))