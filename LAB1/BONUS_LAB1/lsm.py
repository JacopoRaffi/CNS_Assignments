import numpy as np

class LSM:
    '''
    Implementation of Liquid State Machine

    Attributes:
    ----------
    Ne: int
        Number of excitatory neurons
    Ni: int
        Number of inhibitory neurons
    a: np.ndarray
        Time scale of the recovery variable u
    b: np.ndarray
        Sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v
    c: np.ndarray
        After-spike reset value of the membrane potential v
    d: np.ndarray
        After-spike reset of the recovery variable u
    U: np.ndarray
        Scaling of input connections
    S: np.ndarray
        Scaling of recurrent connections
    Wout: np.ndarray
        Output weights
    '''
    def __init__(self, Ne, Ni, win_e, win_i, w_e, w_i):
        '''
        Initialize the Liquid State Machine

        Parameters:
        ----------
        Ne: int
            Number of excitatory neurons
        Ni: int
            Number of inhibitory neurons
        win_e: float
            Scaling of input connections for excitatory neurons
        win_i: float
            Scaling of input connections for inhibitory neurons
        w_e: float
            Scaling of recurrent connections for excitatory neurons
        w_i: float
            Scaling of recurrent connections for inhibitory neurons
        '''
        self.Ne = Ne
        self.Ni = Ni
        
        re = np.random.rand(Ne)
        ri = np.random.rand(Ni)
        self.a = np.concatenate((0.02*np.ones(Ne), 0.02+0.08*ri))
        self.b = np.concatenate((0.2*np.ones(Ne), 0.25-0.05*ri))
        self.c = np.concatenate((-65+15*re**2, -65*np.ones(Ni)))
        self.d = np.concatenate((8-6*re**2, 2*np.ones(Ni)))

        # scaling of input connections
        self.U = np.concatenate((win_e*np.ones(Ne), win_i*np.ones(Ni)))

        # scaling of recurrent connections
        self.S = np.concatenate((w_e*np.random.rand(Ne+Ni, Ne), -w_i*np.random.rand(Ne+Ni, Ni)), axis=1)

        self.Wout = None

    def __compute_states(self, input:np.ndarray):
        '''
        Compute the reservoir states

        Parameters:
        ----------
        input: np.ndarray
            Input data
        
        Returns:
        --------
        return: np.ndarray
            Return the reservoir states
        '''
        v = -65*np.ones(self.Ne + self.Ni)  # Initial values of v
        u = self.b*v  # Initial values of u
        states = []  # here we construct the matrix of reservoir states

        for t in range(len(input)): 
            I = input[t] * self.U
            fired = np.where(v >= 30)[0]  # indices of spikes
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(self.S[:, fired], axis=1)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + self.a*(self.b*v - u)
            states.append(v >= 30)

        return np.array(states, dtype=int)

    def fit(self, input:np.ndarray, target:np.ndarray):
        '''
        Fit the output weights (Wout)

        Parameters:
        ----------
        input: np.ndarray
            Input data
        target: np.ndarray
            Target data
        
        Returns:
        --------
        return: -
        '''
        states = self.__compute_states(input)

        self.Wout = np.linalg.pinv(states) @ target

    def __call__(self, input:np.ndarray):
        '''
        Predict the target values

        Parameters:
        ----------
        input: np.ndarray
            Input data
        
        Returns:
        --------
        return: np.ndarray
            Return the predicted target values
        '''
        states = self.__compute_states(input)

        return (states @ self.Wout)
