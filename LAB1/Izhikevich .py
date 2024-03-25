import numpy

def f(u:float, w:float, I:float):
    '''
        Set hyperparameters for SGD training

        Parameter
        ---------
        u: float
            current membrane potential value 
        w: float
            current recovery variable value
        I: float
            present current input value

        Returns
        -------
        return: float 
            the derivative of the membrane

    '''
    return 0.04 * (u**2) + 5*u + 140 - w + I

def g(u:float, w:float, a:float, b:float):
    '''
        Set hyperparameters for SGD training

        Parameter
        ---------
        u: float
            current membrane potential value 
        w: float
            current recovery variable value
        a: float
            time scale of the recovery variable
        b: float
            sensitivity of the recovery variable to flutuations of the membrane potential
        
        Returns
        -------
        return: float
            the derivative of the recovery variable

    '''
    return a*(b*u - w)

def Izhikevich(u:float, w:float, 
               a:float, b:float, c:float, d:float, 
               I:numpy.ndarray, time_steps:numpy.ndarray, h:float):
    
    '''
        Set hyperparameters for SGD training

        Parameter
        ---------
        u: float
            current membrane potential value 
        w: float
            current recovery variable value
        a: float
            time scale of the recovery variable
        b: float
            sensitivity of the recovery variable to flutuations of the membrane potential
        c: float
            after-spike reset value of the membrane potential
        d: float
            after-spike reset of the recovery variable
        I: numpy array
            values of the current input over time
        time_steps: numpy array
            the time steps to consider during the iteration
        h: float
            time step (used for the Leap-Frog method)

        Returns
        -------
        return: the membrane potential and recovery variable values over time

    '''

    u_values = []
    w_values = []

    for i, _ in enumerate(time_steps):
        # application of Leap-Forg method
        u = u + h * f(u, w, I[i])
        w = w + h * g(a, b, u, w)

        if u >= 30:
            u_values.append(30)
            u = c
            w = w + d
        else:
            u_values.append(u)

        w_values.append(w)

    return u_values, w_values