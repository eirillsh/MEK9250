import numpy as np
import scipy as sp
import matplotlib.pyplot as plt




def u(x, mu):
    return ( np.exp(-x / mu) - 1 ) / ( np.exp(-1 / mu) - 1 )

def ux(x, mu):
    return ( -np.exp(-x / mu) / mu - 1 ) / ( np.exp(-1 / mu) - 1 )

def uxx(x, mu):
    return ( np.exp(-x / mu) / mu**2 - 1 ) / ( np.exp(-1 / mu) - 1 )

def H2norm(mu):
    I, err = sp.integrate.quad(lambda x: u(x, mu)**2 + ux(x, mu)**2 + uxx(x, mu)**2, 0, 1, epsabs=5e-13, epsrel=5e-13)
    return np.sqrt(I)

def main():

    I, err = sp.integrate.quad(lambda x: uxx(x, 0.001)**2, 0, 1, epsabs=5e-13, epsrel=5e-13)

    print(np.sqrt(I), err)

    print(H2norm(0.001))


    xx = np.linspace(0, 1, 501)

    plt.figure()
    plt.plot(xx, u(xx, 0.001), 'k--')
    plt.show()

if __name__ == '__main__':
    main()
