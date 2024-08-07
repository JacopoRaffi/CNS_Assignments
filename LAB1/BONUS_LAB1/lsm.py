import numpy as np

class LSM:
    def __init__(self, Ne, Ni, win_e, win_i, w_e, w_i):
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
        v = -65*np.ones(self.Ne + self.Ni)  # Initial values of v
        u = self.b*v  # Initial values of u
        #firings = []  # spike timings
        states = []  # here we construct the matrix of reservoir states

        for t in range(len(input)): 
            I = input[t] * self.U
            fired = np.where(v >= 30)[0]  # indices of spikes
            #firings.append(np.column_stack((t + np.zeros_like(fired), fired)))
            v[fired] = self.c[fired]
            u[fired] = u[fired] + self.d[fired]
            I = I + np.sum(self.S[:, fired], axis=1)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + self.a*(self.b*v - u)
            states.append(v >= 30)

        #firings = np.concatenate(firings)

        return np.array(states, dtype=int)

    def fit(self, input:np.ndarray, target:np.ndarray, lamd:float = 0.1):
        states = self.__compute_states(input)

        # Tikhonov regularization is applied
        self.Wout = np.linalg.pinv(states) @ target

    def __call__(self, input:np.ndarray):
        states = self.__compute_states(input)

        return (states @ self.Wout)
    

if __name__ == "__main__":
    lsm = LSM(800, 200, 0.1, 0.5, 0.1, 0.5)

    input_data = np.random.rand(10000)  # Replace with your input data
    lsm.fit(input_data, input_data) 

    i = lsm(input_data)
    print(i.shape)

