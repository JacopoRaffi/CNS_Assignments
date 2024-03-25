# TODO: comments

def f(u, w, I):
    return 0.04*(u**2) + 5*u + 140 - w + I

def g(a, b, u, w):
    return a*(b*u - w)

def Izhikevich(u, w, a, b, c, d, I, time_steps, h):
    u_values = []
    w_values = []

    # TODO: timesteps
    u = u + h * f(u, w, I)
    w = w + h * g(a, b, u, w)

    if u >= 30:
        u_values.append(30)
        u = c
        w = w + d
    else:
        u_values.append(30)

    w_values.append(w)

    return u_values, w_values