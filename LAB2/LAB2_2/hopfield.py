import numpy as np

class HopfieldNetwork:
    '''
    Implementation of the Hopfield Network

    Attributes:
    ----------
    data: np.ndarray
        Patterns to be stored in the network
    n_patterns: int
        Number of patterns
    size: int
        number of neurons
    '''

    def __init__(self, data:np.ndarray):
        '''
        Parameters:
        -----------
        data: np.ndarray
            Patterns to be stored in the network

        Returns:
        --------
        return: -
        '''
        self.n_patterns, self.size = data.shape
        self.W = self.__learning(data)
        self.patterns = data
        

    def __learning(self, data:np.ndarray):
        '''
        Learning the weights of the network

        Parameters:
        -----------
        data: np.ndarray
            Patterns to be stored in the network
        
        Returns:
        --------
        return: np.ndarray
            Weight matrix of the network
        '''

        W = np.zeros((self.size, self.size))

        for pattern in data:
            W += np.outer(pattern, pattern)
        
        np.fill_diagonal(W, 0)
        
        return (W / self.size)

    def __energy(self, state:np.ndarray) -> float:
        '''
        Compute the energy of the network

        Parameters:
        -----------
            state: np.ndarray
                State of the network's neurons
        
        Returns:
        --------
            return: float
                Energy of the network
        '''
        
        return -0.5 * (np.dot(np.dot(self.W, state), state))

    def __overlap(self, state:np.ndarray, true_pattern:np.ndarray) -> float:
        '''
        Compute the overlap of the network

        Parameters:
        -----------
        state: np.ndarray
            State of the network's neurons

        true_pattern: np.ndarray
            Target Pattern
        
        Returns:
        --------
        return: float
            Overlap of the network
        '''
        return (1/self.size) * np.dot(state, true_pattern)
    
    def add_pattern(self, pattern:np.ndarray):
        '''
        Add a new pattern to the network

        Parameters:
        -----------
        pattern: np.ndarray
            New pattern to be added

        Returns:
        --------
        return: -
        '''
        self.W += (np.outer(pattern, pattern)) / self.size # incremental update of the weights

    def __call__(self, input:np.ndarray, steps:int = 10, bias = 0.6, true_pattern:np.ndarray = None) -> tuple:
        
        state = input.copy()
        energy_history, overlap_history = [], []
        not_converged = True
        step = 0

        while (step < steps) and (not_converged):
            neurons_order = np.random.permutation(self.size)
            previous_state = state.copy()

            for neuron in neurons_order:
                output = np.dot(self.W[neuron], state) + bias
                state[neuron] = np.where(output > 0, 1, -1)

                energy_history.append(self.__energy(state))
                overlap_history.append(self.__overlap(state, true_pattern))
            
            step += 1
            not_converged = not np.array_equal(state, previous_state)
        
        return state, energy_history, overlap_history