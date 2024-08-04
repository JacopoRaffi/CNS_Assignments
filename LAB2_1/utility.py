import numpy as np
import matplotlib.pyplot as plt

def eig_plot(data:np.ndarray, weights:np.ndarray):
    Q = np.corrcoef(data.T)
    eig_values, eig_vecs = np.linalg.eig(Q)
    i_max_eigval = np.argsort(eig_values)[-1]
    origin = np.array([0,0])
    
    plt.scatter(data[:, 0], data[:, 1])
    plt.quiver(*origin, weights[0], weights[1], color='g', label='weights')
    plt.quiver(*origin, eig_vecs[:, i_max_eigval][0], eig_vecs[:, i_max_eigval][1],  color='r', label='principal eigenvector', scale=10)
    plt.legend()

    plt.show()

def weights_plot():
    pass