import numpy as np

class HopfieldNetwork:
    def __init__(self, data:np.ndarray) -> None:
        self.n_patterns, self.size = data.shape
        self.W = self.__learning(data)
        self.patterns = data
        

    def __learning(self, data:np.ndarray) -> None:
        W = np.zeros((self.size, self.size))

        for pattern in data:
            W += np.outer(pattern, pattern)
        
        np.fill_diagonal(W, 0)
        
        return (W / self.size)

    def __energy(self, state:np.ndarray) -> float:
        return -0.5 * (np.dot(np.dot(self.W, state), state))

    def __overlap(self, state:np.ndarray, true_pattern:np.ndarray) -> float:
        return (1/self.size) * np.dot(state, true_pattern)

    def __call__(self, input:np.ndarray, steps:int = 10, bias = 0.6, 
                 history:bool = False, true_pattern:np.ndarray = None) -> np.ndarray:
        state = input.copy()

        if history:
            assert true_pattern is not None, "True pattern is required for overlap history"

        energy_history, overlap_history = [], []
        not_converged = True
        step = 0

        while (step < steps) and (not_converged):
            neurons_order = np.random.permutation(self.size)
            previous_state = state.copy()

            for neuron in neurons_order:
                output = np.dot(self.W[neuron], state) + bias
                state[neuron] = np.where(output > 0, 1, -1)

                if history:
                    energy_history.append(self.__energy(state))
                    overlap_history.append(self.__overlap(state, true_pattern))
            
            step += 1
            not_converged = not np.array_equal(state, previous_state)
        
        return state, energy_history, overlap_history