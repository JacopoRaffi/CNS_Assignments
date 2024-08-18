import numpy as np
import matplotlib.pyplot as plt
import json

def save_to_json(filename:str, array_list:list):
    '''
    Save the data to a json file

    Parameters
    ---------
    data: list
        data to save
    filename: str
        name of the file

    Returns
    -------
    return: -

    '''

    data = [arr.tolist() for arr in array_list]

    with open(filename, 'w') as f:
        json.dump(data, f)

def load_from_json(filename:str):
    '''
    Load the data from a json file

    Parameters
    ---------
    filename: str
        name of the file

    Returns
    -------
    return: list
        data loaded from the file

    '''

    with open(filename, 'r') as f:
        data = json.load(f)

    return [np.array(arr) for arr in data]


def eig_plot(data:np.ndarray, weights:np.ndarray):
    '''
    Plot the principal eigenvector, the weights vector and the dataset

    Parameters
    ---------
    data: numpy array
         dataset 
    w: numpy array
        final weghts vector

    Returns
    -------
    return: -

    '''
        
    Q = np.corrcoef(data.T)
    eig_values, eig_vecs = np.linalg.eig(Q)
    idx_max_eigval = np.argsort(eig_values)[-1]
    origin = np.array([0,0])
    
    plt.scatter(data[:, 0], data[:, 1])
    plt.quiver(*origin, weights[0], weights[1], color='g', label='weights')
    plt.quiver(*origin, eig_vecs[:, idx_max_eigval][0], eig_vecs[:, idx_max_eigval][1],  color='r', label='principal eigenvector', scale=10)
    plt.legend()

    plt.show()

def weights_plot(w_history:list, norm_history:list):
    '''
    Plot the evolution of the weights and the norm of the weights over the epochs
    
    Parameters
    ---------
    w_history: list
        list of the weights vector at each epoch
    norm_history: list
        list of the norm of the weights vector at each epoch

    Returns
    -------
    return: -
    
    '''

    fig, axs = plt.subplots(1, 3, figsize=(25,6))

    axs[0].plot(np.array(w_history)[:, 0], color='r')
    axs[0].set_title('Weight 1 Evolution')
    axs[0].set_xlabel('Epochs')
    
    axs[1].plot(np.array(w_history)[:, 1], color='g')
    axs[1].set_title('Weight 2 Evolution')
    axs[1].set_xlabel('Epochs')

    axs[2].plot(norm_history)
    axs[2].set_title('Weights Norm Evolution')
    axs[2].set_xlabel('Epochs')

    plt.show()