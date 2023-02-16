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

    left_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=leftfacets)
    right_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=rightfacets)

    left_bc = fem.dirichletbc(ScalarType(0), left_bc_dofs, V)
    right_bc = fem.dirichletbc(ScalarType(1), right_bc_dofs, V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    mu = fem.Constant(domain, ScalarType(mu_val))
    w = fem.Constant(domain, ScalarType((-1,)))

    if SUPG:       
        beta = 0.5
        h = 1 / (N-1)
        vv = v + beta*h * ufl.dot(w, ufl.grad(v))
    else:
        vv = v

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(vv)) * ufl.dx + ufl.dot(ufl.grad(u), w) * vv * ufl.dx
    L = f * vv * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh

mu = 0.001
N = 21

uh_supg = run_problem(N, mu, SUPG=True)
xx_supg = np.copy(uh_supg.function_space.mesh.geometry.x[:,0])
uu_supg = np.copy(uh_supg.vector.array)

uh_cg = run_problem(N, mu, SUPG=False)
xx_cg = np.copy(uh_cg.function_space.mesh.geometry.x[:,0])
uu_cg = np.copy(uh_cg.vector.array)

xx_long = np.linspace(0, 1, 1001)
uu_ex_long = u_func(xx_long, mu)

plt.figure()

plt.plot(xx_supg, uu_supg, 'k-', label=r"$u_\mathrm{supg}$")
plt.plot(xx_cg, uu_cg, 'k--', label=r"$u_\mathrm{cg}$")
plt.plot(xx_long, uu_ex_long, 'k:', label=r"$u_\mathrm{ex}$")

plt.legend()

plt.show()
