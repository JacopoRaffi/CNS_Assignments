import numpy
import matplotlib.pyplot as plt

# functions 'f' and 'g' are written separated for code elegance

def du_dt(u:float, w:float, I:float):
    # compute the derivative of the membrane potential

    return 0.04 * (u**2) + 5*u + 140 - w + I

def dw_dt(u:float, w:float, a:float, b:float):
    # compute the derivative of the recovery variable

    return a*(b*u - w)

def Izhikevich(u:float, w:float, 
               a:float, b:float, c:float, d:float, 
               I:numpy.ndarray, time_steps:numpy.ndarray, h:float,
               f:callable = du_dt, g:callable = dw_dt):
    
    '''
        Compute the discretize version of Izhikevich

        Parameters
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
        f: callable
            derivative of the membrane potential
        g: callable
            derivative of the recovery variable

        Returns
        -------
        return: tuple 
            the membrane potential and recovery variable values over time

    '''

    u_values = [u]
    w_values = [w]

    for i, _ in enumerate(time_steps):
        # application of Leap-Forg method
        u = u + h * f(u, w, I[i])
        w = w + h * g(u, w, a, b)

        if u >= 30:
            u_values.append(30)
            u = c
            w = w + d
        else:
            u_values.append(u)

        w_values.append(w)

    return u_values, w_values

def show_charts(u_values:list, w_values:list, I: numpy.array, time_steps:list, scale:float = 1.):
    '''
        simply shows the membrane dynamics and the phase portrait plots

        Parameters
        ---------
        u_values: list
            membrane potential's values over time
        w_values: list
            recovery variable's values over time
        I: numpy array
            values of the current input over time
        time_steps: numpy array
            the time steps to consider during the iteration

        Returns
        -------
        return: -
    '''

    fig, ax = plt.subplots(1,2, figsize=(20,5))
    I = numpy.array(I)
    max_i = max(scale*I)
    min_u = min(u_values)
    I = [i - (abs(max_i - min_u)) - 10  for i in scale*I]

    ax[0].set_xlabel("Time (t)")
    ax[0].set_ylabel("Membrane potential (u)")
    ax[0].set_title("Membrane dynamics")

    ax[1].set_xlabel("Membrane potential (u)")
    ax[1].set_ylabel("Recovery variable (w)")
    ax[1].set_title("Phase portrait")
    
    ax[0].plot(time_steps, u_values[1:])
    ax[0].plot(time_steps, I, linestyle='--') # I is the input current
    ax[1].plot(u_values, w_values)