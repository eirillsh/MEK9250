import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def phi_builder(mesh: np.ndarray, k):

    assert len(mesh.shape) == 1 and mesh.shape[0] >= 2

    N = mesh.shape[0] - 1
    
    if k == 0:
        def phi(x):
            x0, x1 = np.copy(mesh[0]), np.copy(mesh[1])

            return np.where(np.logical_and(x >= x0, x < x1), (x1 - x) / (x1 - x0), 0.0*x)
        
        return phi
    
    elif k == N:
        def phi(x):
            xN, xNm = np.copy(mesh[-1]), np.copy(mesh[-2])

            return np.where(np.logical_and(x > xNm, x <= xN), (x - xNm) / (xN - xNm), 0.0*x)
            
        return phi
    
    elif 0 < k < N:
        def phi(x):
            xkm, xk, xkp = np.copy(mesh[k-1]), np.copy(mesh[k]), np.copy(mesh[k+1])

            return np.where(np.logical_and(x > xkm, x <= xk), (x - xkm) / (xk - xkm), 0.0*x) + \
                   np.where(np.logical_and(x > xk, x < xkp), (xkp - x) / (xkp - xk), 0.0*x)
        
        return phi
    
def dphi_builder(mesh: np.ndarray, k):

    assert len(mesh.shape) == 1 and mesh.shape[0] >= 2

    N = mesh.shape[0] - 1
    
    if k == 0:
        def dphi(x):
            x0, x1 = np.copy(mesh[0]), np.copy(mesh[1])

            return np.where(np.logical_and(x >= x0, x < x1), (0.0*x - 1.0) / (x1 - x0), 0.0*x)
        
        return dphi
    
    elif k == N:
        def dphi(x):
            xN, xNm = np.copy(mesh[-1]), np.copy(mesh[-2])

            return np.where(np.logical_and(x > xNm, x <= xN), (0.0*x + 1.0) / (xN - xNm), 0.0*x)
            
        return dphi
    
    elif 0 < k < N:
        def dphi(x):
            xkm, xk, xkp = np.copy(mesh[k-1]), np.copy(mesh[k]), np.copy(mesh[k+1])

            return np.where(np.logical_and(x > xkm, x <= xk), (0.0*x + 1.0) / (xk - xkm), 0.0*x) + \
                   np.where(np.logical_and(x > xk, x < xkp), (0.0*x - 1.0) / (xkp - xk), 0.0*x)
        
        return dphi


N = 5
mesh = np.linspace(0, 1, N)

# xx = np.linspace(0, 1, 201)
# plt.figure()

# for k in range(N):
#     phi = phi_builder(mesh, k)
#     plt.plot(xx, phi(xx))

# plt.figure()

# for k in range(N):
#     dphi = dphi_builder(mesh, k)
#     plt.plot(xx, dphi(xx))


def quad(u, start=0, stop=1, tol=1e-13):

    return sp.integrate.quad(u, start, stop, epsabs=tol)[0]

def quad2(u, v, start=0, stop=1, tol=1e-13):

    return sp.integrate.quad(lambda x: u(x)*v(x), start, stop, epsabs=tol)[0]


def N_builder(mesh, k):
    return phi_builder(mesh, k)
def dN_builder(mesh, k):
    return dphi_builder(mesh, k)

def L_builder(mesh, k):
    beta = 0.5
    h = mesh[1] - mesh[0]
    N = N_builder(mesh, k)
    dN = dN_builder(mesh, k)
    return lambda x: N(x) - beta*h * dN(x)



def dL_builder(mesh, k):
    return dN_builder(mesh, k) # The corrective part disappears when differentiating for CG-1 elements


def a(u, v, du, dv, mu, start=0.0, stop=1.0):
    return mu * quad2(du, dv, start=start, stop=stop) - quad2(du, v, start=start, stop=stop)

def l(v, mu, start=0.0, stop=1.0):
    #f = lambda x: mu*np.pi**2*np.sin(np.pi*x) - np.pi*np.cos(np.pi*x)
    f = lambda x: 0*x
    return quad2(f, v, start=start, stop=stop)

# L_builder = N_builder
N = 101
mesh = np.linspace(0, 1, N)
mu = 0.001
bc0 = 0.0
bc1 = 1.0

A = np.zeros((N, N))
b = np.zeros((N,))

# u(x) = u^j N_j(x)
# v(x) = v^i L_i(x)

# v^T A u = A_ij u^j v^i
# A_ij = a(N_j, L_i)
# v^T b = b_i v^i
# b_i = l(L_i)

for j in range(1,N-1):
    for i in range(1,N-1):
        if abs(i - j) > 2:
            continue
        intstart = mesh[min(i,j)-1]
        intstop = mesh[max(i,j)+1]

        A[i,j] = a(N_builder(mesh, j), L_builder(mesh, i),
                   dN_builder(mesh, j), dL_builder(mesh, i), mu, 
                   start=intstart, stop=intstop)
        b[i] = l(L_builder(mesh, i), mu,
                 start=intstart, stop=intstop)

A[1,0] = a(N_builder(mesh, 0), L_builder(mesh, 1), 
           dN_builder(mesh, 0), dL_builder(mesh, 1), mu,
           start=0, stop=mesh[2])
A[-2,-1] = a(N_builder(mesh, len(mesh)-1), L_builder(mesh, len(mesh)-2), 
           dN_builder(mesh, len(mesh)-1), dL_builder(mesh, len(mesh)-2), mu,
           start=mesh[-3], stop=mesh[-1])

A[0,:] = 0.0
A[0,0] = 1.0
b[0] = bc0

A[-1,:] = 0.0
A[-1,-1] = 1.0
b[-1] = bc1

u = np.linalg.solve(A, b)

print(A)

plt.figure()

plt.plot(mesh, u, 'k-', label="$u_h$")
# plt.plot(mesh, np.sin(np.pi*mesh), label="$u_e$")
plt.legend()

plt.show()


