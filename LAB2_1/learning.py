import numpy as np

class ABCRule:
    def __init__(self, input_size:int, lr:float = 1e-4):
        RANDOM_STATE = 42
        rand_gen = np.random.default_rng(RANDOM_STATE)
        self.w = rand_gen.uniform(-1, 1, input_size) # input is 2-dimension
        self.lr = lr

    def predict(self, u:np.ndarray) -> np.ndarray:
        return np.inner(self.w, u)

    def update(self, u:np.ndarray, v:np.ndarray) -> np.ndarray:
        pass


def train(rule:ABCRule, data:np.ndarray, 
          threshold:float = 1e-5, epochs:int = 100) -> tuple:
   
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
    def __init__(self, input_size:int, lr:float = 1e-4):
        super().__init__(input_size, lr)

    def update(self, u:np.ndarray, v:np.ndarray) :
        delta_w = self.lr * v * u # compute the delta W
        self.w += delta_w # update the weights


class OjaRule(ABCRule):
    def __init__(self, input_size:int, lr:float = 1e-4, alpha:float = 2):
        super().__init__(input_size, lr)
        self.alpha = alpha

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        delta_w = self.lr * ((v * u) - self.alpha * (np.power(v, 2) * self.w))
        self.w += delta_w

class SNRule(ABCRule):
    def __init__(self, input_size:int, lr:float = 1e-4):
        super().__init__(input_size, lr)

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = np.ones_like(u)
        n_u = len(u)

        delta_w = self.lr * (v * (u - (np.inner(u, n)) * n / n_u))
        self.w += delta_w

class CovarianceRule(ABCRule):
    def __init__(self, input_size:int, lr:float = 1e-4, theta:float = 1):
        super().__init__(input_size, lr)
        self.theta = theta

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        delta_w = self.lr * (v * (u - self.theta))
        self.w += delta_w

class BCMRule(ABCRule):
    def __init__(self, input_size:int, lr:float = 1e-4, theta:float = 0.1, theta_lr:float = 0.001):
        super().__init__(input_size, lr)
        self.theta = theta  
        self.theta_lr = theta_lr

    def __theta_update(self, v:np.ndarray):
        self.theta += self.theta_lr * (np.power(v, 2) - self.theta)

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.__theta_update(v)
        delta_w = self.lr * (v * u * (v - self.theta))
        self.w += delta_w