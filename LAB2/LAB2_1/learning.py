import numpy as np

class ABCRule:
    '''
    Abstract Class for the learning rules
    
    Attributes
    ----------
    w : numpy array
        weights of the neuron
    lr : float
        learning rate for training
    '''

    def __init__(self, input_size:int, lr:float = 1e-4):
        '''
        Parameters
        ---------
        input_size : int
            size of the input vector
        lr : float
            learning rate for training
        '''

        RANDOM_STATE = 42
        rand_gen = np.random.default_rng(RANDOM_STATE)
        self.w = rand_gen.uniform(-1, 1, input_size) # input is 2-dimension
        self.lr = lr

    def __call__(self, u:np.ndarray) -> np.ndarray:
        '''
        Compute the output of the neuron

        Parameters
        ---------
        u : numpy array
            input vector
        '''

        return np.inner(self.w, u)

    def update(self, u:np.ndarray, v:np.ndarray) -> np.ndarray:
        '''
        Update the weights of the neuron (every sub-class needs to implement this method)

        Parameters
        ---------
        u : numpy array
            input vector
        v : numpy array
            output of the neuron

        Returns
        -------
        return: -
        '''
        pass


def train(rule:ABCRule, data:np.ndarray, 
          threshold:float = 1e-5, epochs:int = 100) -> tuple:
    '''
    Train the neuron with the given rule

    Parameters
    ---------
    rule : ABCRule
        learning rule to use
    data : numpy array
        dataset
    threshold : float
        threshold used to stop the training loop confronted with the norm of the old and new weights
    epochs : int
        max number of epochs

    Returns
    -------
    return: tuple
        tuple containing the history of the weights vector and the norm of the weights vector over the epochs
    '''
   
    norm_distance = np.inf
    w_history = [rule.w.copy()] 
    norm_history = []
    epoch = 0

    while (epoch < epochs) and (norm_distance > threshold):
        np.random.shuffle(data) # shuffle data at each epoch
        w_old = rule.w.copy() # copy the old weights to compare with the new using the norm 2

        for u in data:
            v = rule(u) 
            rule.update(u, v) 

            w_history.append(rule.w.copy())
            norm_history.append(np.linalg.norm(rule.w))
        
        norm_distance = np.linalg.norm(w_old - rule.w)
        epoch += 1

    return w_history, norm_history


# --------------------------------- Implementation of different Update Rules --------------------------------- #
class HebbRule(ABCRule):
    '''
    Basic Hebb Learning Rule

    Attributes
    ----------
    same as ABCRule
    '''
    
    def __init__(self, input_size:int, lr:float = 1e-4):
        super().__init__(input_size, lr)

    def update(self, u:np.ndarray, v:np.ndarray) :
        delta_w = self.lr * v * u 
        self.w += delta_w 


class OjaRule(ABCRule):
    '''
    Oja Learning Rule

    Attributes
    ----------
    same as ABCRule

    alpha : float
        hyperparameter for the Oja rule
    '''
    
    def __init__(self, input_size:int, lr:float = 1e-4, alpha:float = 2):
        super().__init__(input_size, lr)
        self.alpha = alpha

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        delta_w = self.lr * ((v * u) - self.alpha * (np.power(v, 2) * self.w))
        self.w += delta_w

class SNRule(ABCRule):
    '''
    Subtractive Normalization Learning Rule

    Attributes
    ----------
    same as ABCRule
    '''
    
    def __init__(self, input_size:int, lr:float = 1e-4):
        super().__init__(input_size, lr)

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = np.ones_like(u)
        n_u = len(u)

        delta_w = self.lr * (v * (u - (np.inner(u, n)) * n / n_u))
        self.w += delta_w

class CovarianceRule(ABCRule):
    '''
    Covariance Learning Rule

    Attributes
    ----------
    same as ABCRule

    theta : float
        threshold on the presynaptic rate
    '''
    
    def __init__(self, input_size:int, lr:float = 1e-4, theta:float = 1):
        super().__init__(input_size, lr)
        self.theta = theta

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        delta_w = self.lr * (v * (u - self.theta))
        self.w += delta_w

class BCMRule(ABCRule):
    '''
    BCM Learning Rule

    Attributes
    ----------
    same as ABCRule

    theta : float
        varying threshold with its own update 
    theta_lr : float
        learning rate for the threshold
    '''
    
    def __init__(self, input_size:int, lr:float = 1e-4, theta:float = 0.1, theta_lr:float = 0.001):
        super().__init__(input_size, lr)
        self.theta = theta  
        self.theta_lr = theta_lr

    def __theta_update(self, v:np.ndarray):
        '''
        Update the threshold theta

        Parameters
        ---------
        v : numpy array
            output of the neuron

        Returns
        -------
        return: -
        '''
        
        self.theta += self.theta_lr * (np.power(v, 2) - self.theta)

    def update(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.__theta_update(v)
        delta_w = self.lr * (v * u * (v - self.theta))
        self.w += delta_w