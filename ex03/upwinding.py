import numpy as np
import matplotlib.pyplot as plt

MU = 0.001
U0 = 0.0
U1 = 0.0
N = 21
xx = np.linspace(0, 1, N)

"""
    ```
        -mu(x) u''(x) + v(x) u'(x) = f(x)
        u(x) = sin(pi * x)
        u'(x) = pi*cos(pi * x)
        u''(x) = -pi**2 * sin(pi*x)

        f(x) = pi**2 * mu(x) * sin(pi*x) + pi * v(x) * cos(pi*x)
    ```
"""

def v(x):
    return -0.5 + x

def mu(x):
    return 0*x + MU

def f(x):
    return np.pi**2 * mu(x) * np.sin(np.pi*x) + \
           np.pi * v(x) * np.cos(np.pi*x)


def construct(x, v, mu, f, upwinding=True):
    h = x[1] - x[0]

    A = np.zeros((x.shape[0], x.shape[0]), dtype=float)
    A[0,0] = 1.0
    A[-1,-1] = 1.0

    F = f(x)
    F[0] = U0
    F[-1] = U1

    for k, xk in enumerate(x[1:-1], start=1):
        A[k,k] += 2.0 * mu(xk) / h**2
        A[k,k-1] += -1.0 * mu(xk) / h**2
        A[k,k+1] += -1.0 * mu(xk) / h**2

        if upwinding:
            if v(xk) > 0:
                A[k,k+1] += v(xk) / h
                A[k,k] += -v(xk) / h
            else:
                A[k,k] += v(xk) / h
                A[k,k-1] += -v(xk) / h
            
        else:
            A[k,k+1] += v(xk) / (2*h)
            A[k,k+1] += -v(xk) / (2*h)

    return A, F

A, F = construct(xx, v, mu, f, upwinding=True)

U = np.linalg.solve(A, F)


plt.figure()
plt.matshow(np.block([A,F[:,None]]), fignum=0)
plt.colorbar()

plt.figure()
plt.plot(xx, U, 'k-')

plt.show()

