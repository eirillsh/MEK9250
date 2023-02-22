
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl


def compute_H10_norm(uh):

    H1_sqr_form = fem.form( ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H10_error(uh, uex):
    
    H1_sqr_form = fem.form( ufl.dot(ufl.grad(uh-uex), ufl.grad(uh-uex)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H1_norm(uh):

    H1_sqr_form = fem.form( ufl.dot(uh, uh) * ufl.dx + \
                            ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H1_error(uh, uex):
    
    H1_sqr_form = fem.form( ufl.dot(uh-uex, uh-uex) * ufl.dx + \
                            ufl.dot(ufl.grad(uh-uex), ufl.grad(uh-uex)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_L2_norm(uh):

    L2_sqr_form = fem.form( ufl.inner(uh, uh) * ufl.dx )
    L2_local = fem.assemble_scalar(L2_sqr_form)
    L2 = np.sqrt(uh.function_space.mesh.comm.allreduce(L2_local, op=MPI.SUM))

    return L2

def compute_L2_error(uh, uex):

    L2_sqr_form = fem.form( ufl.dot(uh-uex, uh-uex) * ufl.dx )
    L2_local = fem.assemble_scalar(L2_sqr_form)
    L2 = np.sqrt(uh.function_space.mesh.comm.allreduce(L2_local, op=MPI.SUM))

    return L2

def compute_sd_norm(uh, v, mu, h):

    gu = ufl.grad(uh)
    v_d_gu = ufl.dot(v, ufl.grad(gu))

    sd_sqr_form = fem.form( h * v_d_gu*v_d_gu * ufl.dx + mu * gu*gu * ufl.dx )
    sd_local = fem.assemble_scalar(sd_sqr_form)
    sd = np.sqrt(uh.function_space.mesh.comm.allreduce(sd_local, op=MPI.SUM))

    return sd

def compute_sd_error(uh, uex, v, mu, h):

    gu = ufl.grad(uh - uex)
    v_d_gu = ufl.dot(v, ufl.grad(gu))

    sd_sqr_form = fem.form( h * v_d_gu*v_d_gu * ufl.dx + mu * gu*gu * ufl.dx )
    sd_local = fem.assemble_scalar(sd_sqr_form)
    sd = np.sqrt(uh.function_space.mesh.comm.allreduce(sd_local, op=MPI.SUM))

    return sd

def run_problem(N, mu_val, SUPG=True):

    domain = mesh.create_unit_interval(MPI.COMM_SELF, N)

    V = fem.FunctionSpace(domain, ("CG", 1))


    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    leftfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 0.0))
    rightfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 1.0))
    

    left_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, 
                                               entities=leftfacets)
    right_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, 
                                                entities=rightfacets)

    left_bc = fem.dirichletbc(ScalarType(0.0), left_bc_dofs, V)
    right_bc = fem.dirichletbc(ScalarType(0.0), right_bc_dofs, V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)


    mu = fem.Constant(domain, ScalarType(mu_val))
    w = fem.Constant(domain, ScalarType((-1,)))

    u_ex = ufl.sin(ufl.pi * x[0])
    f = ufl.pi**2 * mu * ufl.sin(ufl.pi * x[0]) - ufl.pi * ufl.cos(ufl.pi * x[0])


    if SUPG:       
        beta = 0.5
        h = 1 / (N-1)
        vv = v + beta*h * ufl.dot(w, ufl.grad(v))
    else:
        vv = v

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(vv)) * ufl.dx \
        + ufl.dot(ufl.grad(u), w) * vv * ufl.dx
    L = f * vv * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], 
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    h = 1.0 / N

    gu = ufl.grad(uh - u_ex)
    w_d_gu = ufl.dot(w, ufl.nabla_grad(gu))
    sd_sqr_form = fem.form( h * ufl.dot(w_d_gu,w_d_gu) * ufl.dx + mu * ufl.dot(gu,gu) * ufl.dx )
    sd_local = fem.assemble_scalar(sd_sqr_form)
    sd = np.sqrt(uh.function_space.mesh.comm.allreduce(sd_local, op=MPI.SUM))

    h10_sqr_form = fem.form( ufl.dot(gu, gu) * ufl.dx )
    h10_local = fem.assemble_scalar(h10_sqr_form)
    h10 = np.sqrt(uh.function_space.mesh.comm.allreduce(h10_local, op=MPI.SUM))

    return uh, h10, sd

mu = 0.0000001
N = 10

def u_func(x):
    return np.sin(np.pi*x[0])

uh_supg, h10, sd = run_problem(N, mu, SUPG=True)
xx_supg = np.copy(uh_supg.function_space.mesh.geometry.x[:,0])
uu_supg = np.copy(uh_supg.vector.array)

uh_cg, h10, sd = run_problem(N, mu, SUPG=False)
xx_cg = np.copy(uh_cg.function_space.mesh.geometry.x[:,0])
uu_cg = np.copy(uh_cg.vector.array)

xx_long = np.linspace(0, 1, 1001)
uu_ex_long = u_func(xx_long[None,:])

plt.figure()

plt.plot(xx_supg, uu_supg, 'k-', label=r"$u_\mathrm{supg}$")
plt.plot(xx_cg, uu_cg, 'k--', label=r"$u_\mathrm{cg}$")
plt.plot(xx_long, uu_ex_long, 'k:', label=r"$u_\mathrm{ex}$")

plt.legend()



mus = [1e-3, 1e-4, 1e-5]
Ns = [10 * 2**k for k in range(1, 10)]

data = np.zeros((len(mus), len(Ns), 2))
for i, mu in enumerate(mus):
    for j, N in enumerate(Ns):
        uh, h10, sd = run_problem(N, mu, SUPG=True)
        data[i,j,:] = [h10+0.0, sd+0.0]

hs = 1 / (np.array(Ns))
fig, axs = plt.subplots(1,2)
for i, mu in enumerate(mus):
    axs[0].loglog(hs, data[i,:,0], label=f"${mu=}, H^1_0$")
    axs[1].loglog(hs, data[i,:,1], label=f"${mu=}, "+r"\mathrm{sd}$")

axs[0].legend()
axs[1].legend()
fig.suptitle("w/ SUPG")

data = np.zeros((len(mus), len(Ns), 2))
for i, mu in enumerate(mus):
    for j, N in enumerate(Ns):
        uh, h10, sd = run_problem(N, mu, SUPG=False)
        data[i,j,:] = [h10+0.0, sd+0.0]

hs = 1 / (np.array(Ns))
fig, axs = plt.subplots(1,2)
for i, mu in enumerate(mus):
    axs[0].loglog(hs, data[i,:,0], label=f"${mu=}, H^1_0$")
    axs[1].loglog(hs, data[i,:,1], label=f"${mu=}, "+r"\mathrm{sd}$")

axs[0].legend()
axs[1].legend()
fig.suptitle("w/o SUPG")


plt.show()
