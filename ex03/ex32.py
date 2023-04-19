import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from h2norm import u as u_func, H2norm

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl


def compute_H1_norm(uh):

    H1_sqr_form = fem.form( ufl.dot(uh, uh) * ufl.dx + \
                            ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H1_error(uh, uex):
    #TODO: Power raise
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

def run_problem(N, mu_val):

    domain = mesh.create_unit_interval(MPI.COMM_SELF, N)

    V = fem.FunctionSpace(domain, ("CG", 1))


    domain.topology.create_connectivity(0, 1)

    leftfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 0.0))
    rightfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 1.0))

    left_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=leftfacets)
    right_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=rightfacets)

    left_bc = fem.dirichletbc(ScalarType(0), left_bc_dofs, V)
    right_bc = fem.dirichletbc(ScalarType(1), right_bc_dofs, V)

    # print("Left BC: ", left_bc, left_bc_dofs)
    # print("Right BC:", right_bc, right_bc_dofs)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    mu = fem.Constant(domain, ScalarType(mu_val))
    w = fem.Constant(domain, ScalarType((-1,)))

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(ufl.grad(u), w) * v * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    V2 = fem.FunctionSpace(domain, ("CG", 2))
    uex = fem.Function(V2)
    uex.interpolate(lambda x: u_func(x[0], mu.value))


    print(f"{N=}, mu={mu.value}")
    H1, L2 = compute_H1_error(uh, uex), compute_L2_error(uh, uex)
    print(f"H1 = {H1}")
    print(f"L2 = {L2}")

    uu = np.copy(uh.vector.array)

    del domain, V, V2

    return uu, H1, L2

mu = 0.005
N = 40
# mu = 1e-5
# N = 20000
uu, H1, L2 = run_problem(N, mu)
xx = np.linspace(0, 1, uu.shape[0])

xx_long = np.linspace(0, 1, max(1001, xx.shape[0]))
uu_ex_long = u_func(xx_long, mu)

plt.figure()

plt.plot(xx, uu, 'k-', label="$u_h$")
plt.plot(xx_long, uu_ex_long, 'k:', label="$u_e$")

plt.legend()

mus = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
Ns = [21, 41, 81, 161, 321, 641, 1281, 2561, 5121, 10480, 20562]
Ns = [10 * 2**k for k in range(1, 15)]

data = np.zeros((len(mus), len(Ns), 2))

for i, mu in enumerate(mus):
    for j, N in enumerate(Ns):
        uu, H1, L2 = run_problem(N, mu)
        data[i,j,:] = [H1+0.0, L2+0.0]

print(data) 

hs = 1 / (np.array(Ns) - 1)

fig, axs = plt.subplots(1,2)

for i, mu in enumerate(mus):
    axs[0].loglog(hs, data[i,:,0] / H2norm(mu), label=f"${mu=}, H^1$")
    axs[1].loglog(hs, data[i,:,1] / H2norm(mu), label=f"${mu=}, L^2$")

axs[0].legend()
axs[1].legend()

# axs[0].set_xlim()


print()

plt.show()
