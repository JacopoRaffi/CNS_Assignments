import numpy as np

class HopfieldNetwork:
    def __init__(self, data:np.ndarray) -> None:
        self.n_patterns, self.dim = data.shape
        self.W = self.__learning(data)
        

    def __learning(self, data:np.ndarray) -> None:
        W = np.zeros((self.dim, self.dim))

        for pattern in data:
            W += np.outer(pattern, pattern)
        
        np.fill_diagonal(W, 0)
        
        return (W / self.dim)

    def retrieval(self):
        pass