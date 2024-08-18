import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distort_image(im, prop):
    '''
    Distorts an image by flipping a proportion of its elements

    Parameters:
    ----------
    im: numpy.array
        The image to be distorted
    prop: float
        The proportion of elements to be distorted. It must be a float between 0 and 1
    
    Returns:
    -------
    return: numpy.array
        The distorted image
    '''
    if prop < 0 or prop > 1:
        print('Out-of-bound proportion: going to default 0.05')
        prop = 0.05  # Default
    
    # Calculation of the number of elements to be distorted.
    total_elements = im.size
    num_to_distort = round(total_elements * prop)
    
    # Random selection of the indices of the elements to be distorted.
    indices_to_distort = np.random.permutation(total_elements)[:num_to_distort]
    
    # Flattening the image to facilitate access to and editing of elements.
    im_flat = im.flatten()
    im_flat[indices_to_distort] = -im_flat[indices_to_distort]
    
    # Return of the image to its original format after distortion.
    return im_flat.reshape(im.shape)

def reshape_vector_to_image(csv_filename):
    '''
    Reshapes a vector to a 32x32 image.

    Parameters:
    ----------
    csv_filename: str
        The name of the csv file containing the vector to be reshaped
    
    Returns:
    -------
    return: numpy.array
        The reshaped image
    '''
    p_df = pd.read_csv(csv_filename, header=None)
    p = p_df.to_numpy().reshape(-1) # reshape to have a 1D array
    
    return p.reshape(32, 32).T # reshape to have a 32x32 image

def plot_images(original_image, noisy_image, retrieved_image, noise):
    '''
    Plots the original image, the noisy image and the retrieved image side by side.

    Parameters:
    ----------
    original_image: numpy.array
        The original image
    noisy_image: numpy.array
        The noisy image
    retrieved_image: numpy.array
        The retrieved image by the hopfield network
    noise: int
        The noise level of the noisy image
    
    Returns:
    -------
    return: -
    '''
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    mse = np.mean(np.square(original_image.reshape(-1) - retrieved_image.reshape(-1))) # reshape(-1) gives a flatten view of the array

    fig.suptitle(f'Image Comparison - MSE: {mse}')

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')

    axs[1].imshow(noisy_image, cmap='gray')
    axs[1].set_title(f'Corrupted Image (Noise: {noise})')
    
    axs[2].imshow(retrieved_image, cmap='gray')
    axs[2].set_title('Retrieved Image')

def plot_history(energy_history:list, overlap_history:list, noise:int):
    '''
    Plots the history of the energy and overlap of the hopfield network.

    Parameters:
    ----------
    energy_history: list
        The history of the energy of the hopfield network
    overlap_history: list
        The history of the overlap of the hopfield network
    noise: int
        The noise level of the noisy image
    
    Returns:
    -------
    return: -
    '''
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle(f'History of Energy and Overlap - Noise: {noise}')

    axs[0].plot(energy_history)
    axs[0].set_title('Energy History')
    axs[0].set_xlabel('Updates')
    axs[0].set_ylabel('Energy')

    axs[1].plot(overlap_history, color='g')
    axs[1].set_title('Overlap History')
    axs[1].set_xlabel('Updates')
    axs[1].set_ylabel('Overlap')