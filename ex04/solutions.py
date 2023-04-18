import numpy as np


class poiseuille:
    """
        Exact solution of stationary Poiseuille flow with
        flipped sign of pressure.
    """
    def u(x):
        shape = (2, *x.shape[1:])
        values = np.zeros(shape)
        values[0] = x[1]*(1.0 - x[1])
        return values
    def p(x):
        return x[0] * 2.0 - 2.0
    


