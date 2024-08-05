import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distort_image(im, prop):
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
    p_df = pd.read_csv(csv_filename, header=None)
    p = p_df.to_numpy().reshape(-1) # reshape to have a 1D array
    
    return p.reshape(32, 32).T # reshape to have a 32x32 image

def plot_images(original_image, noisy_image, retrieved_image, noise):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    mse = np.mean(np.square(original_image.reshape(-1) - retrieved_image.reshape(-1))) # reshape(-1) gives a flatten view of the array

    fig.suptitle(f'Image Comparison - MSE: {mse}')

    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Original Image')

    axs[1].imshow(noisy_image, cmap='gray')
    axs[1].set_title(f'Corrupted Image (Noise: {noise})')
    
    axs[2].imshow(retrieved_image, cmap='gray')
    axs[2].set_title('Retrieved Image')

def plot_history(energy_history:list, overlap_history:list):
    pass