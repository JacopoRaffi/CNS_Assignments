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

def weights_plot(w_history:np.ndarray, norm_history:np.ndarray):
    fig, axs = plt.subplots(1, 3, figsize=(25,6))

    axs[0].plot(np.array(w_history)[:, 0], color='r')
    axs[0].set_title('Weight 1 Evolution')
    axs[0].set_xlabel('Epochs')
    
    axs[1].plot(np.array(w_history)[:, 1], color='g')
    axs[1].set_title('Weight 2 Evolution')
    axs[1].set_xlabel('Epochs')

    axs[2].plot(np.power(norm_history, 2))
    axs[2].set_title('Weights Norm Evolution')
    axs[2].set_xlabel('Epochs')

    plt.show()