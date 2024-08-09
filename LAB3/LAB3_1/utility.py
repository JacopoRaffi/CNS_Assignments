import matplotlib.pyplot as plt

def plot_loss(training_loss:list, test_loss:list, test:bool = False):
    '''
    Plot the training and evaluation loss history

    Parameters:
    ----------
    training_loss: list
        The training loss history
    test_loss: list
        The evaluation loss history (it could be the test loss history or the validation loss history)
    test: bool
        Whether the loss history is the test loss history or the validation loss history (True for test loss history)

    Returns:
    -------
    return: -
    '''
    # change the label considering if it is the test loss history or the validation loss history
    val_label = 'Test MSE' if test else 'Validation MSE'

    plt.figure(figsize=(12, 6))
    plt.title(f'Training MSE: {round(training_loss[-1], 5)} - {val_label}: {round(test_loss[-1], 5)}')
    plt.plot(training_loss, label='Training Loss')
    plt.plot(test_loss, label=val_label)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)') 
    plt.legend()
    plt.show()