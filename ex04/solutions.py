import numpy as np


class poiseuille:

    def u(x):
        # values = np.zeros_like(x[:2,...])
        # shape = list(x.shape)
        # shape[0] = 2
        shape = (2, *x.shape[1:])
        values = np.zeros(shape)
        values[0] = x[1]*(1.0 - x[1])
        return values
    def p(x):
        return x[0] * 2.0 - 2.0
    


