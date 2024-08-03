import numpy as np

class ABCRule:
    def __init__(self, input_size:int, eta:float = 1e-4) -> None:
        RANDOM_STATE = 42
        rand_gen = np.random.default_rng(RANDOM_STATE)
        self.w = rand_gen.uniform(-1, 1, input_size) # input is 2-dimension
        self.eta = eta

    def predict(self, u:np.ndarray) -> np.ndarray:
        return np.inner(self.w, u)

    def update(self, u:np.ndarray, v:np.ndarray) -> np.ndarray:
        pass


def train(rule:ABCRule, data:np.ndarray, 
          threshold:float = 1e-8, epochs:int = 1000) -> tuple:
   
    norm_distance = np.inf
    w_history = [rule.w.copy()]
    norm_history = []
    epoch = 0

    while (epoch < epochs) and (norm_distance > threshold):
        np.random.shuffle(data) # shuffle data at each epoch
        w_old = rule.w.copy() # copy the old wiehts to compare with the new using the norm2

        for u in data:
            v = rule.predict(u)
            rule.update(u, v)

            w_history.append(rule.w.copy())
            norm_history.append(np.linalg.norm(rule.w))
        
        norm_distance = np.linalg.norm(w_old - rule.w)
        epoch += 1

    return w_history, norm_history


# --------------------------------- Implementation of different Rules --------------------------------- #
class HebbRule(ABCRule):
    def __init__(self, input_size:int, eta:float = 1e-4) -> None:
        super().__init__(input_size, eta)

    def update(self, u:np.ndarray, v:np.ndarray) -> None:
        delta_w = self.eta * v * u # compute the delta W
        self.w += delta_w # update the weights


class OjaRule(ABCRule):
    def __init__(self, input_size:int, eta:float = 1e-4, alpha:float = 2) -> None:
        super().__init__(input_size, eta)
        self.alpha = alpha

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        delta_w = self.eta * ((v * u) - self.alpha * (np.power(v, 2) * self.w))
        self.w += delta_w