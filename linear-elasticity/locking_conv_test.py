# mpirun -np 4 -genv OMP_NUM_THREADS=1 python3 locking_conv_test.py

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem



def run_problem(N, mu_val=1.0, lambda_val=1.0e0, p=1):

    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                 cell_type=mesh.CellType.triangle)


    V = fem.VectorFunctionSpace(domain, ("CG", p))

    mu = fem.Constant(domain, ScalarType(mu_val))
    lambda_ = fem.Constant(domain, ScalarType(lambda_val))

    x = ufl.SpatialCoordinate(domain)

    phi = ufl.sin(ufl.pi * x[0] * x[1])
    u_ex = ufl.as_vector((phi.dx(1), -phi.dx(0)))

    lapl_u_ex_1 =  phi.dx(1).dx(0).dx(0) + phi.dx(1).dx(1).dx(1)
    lapl_u_ex_2 = -phi.dx(0).dx(0).dx(0) - phi.dx(0).dx(1).dx(1)

    f = -mu * ufl.as_vector((lapl_u_ex_1, lapl_u_ex_2))

    """ Boundary conditions """
    expr = fem.Expression(u_ex, V.element.interpolation_points())
    uD = fem.Function(V)
    uD.interpolate(expr)

    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    a = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        lambda_ * ufl.div(u) * ufl.div(v) * ufl.dx
    l = ufl.inner(f, v) * ufl.dx

    problem = fem.petsc.LinearProblem(a, l, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    uh = problem.solve()
    uh.name = "u_h"

    uD.name = "u_exact"


    order_increase = 2
    Vp = fem.VectorFunctionSpace(domain, ("CG", p+order_increase))
    uhp = fem.Function(Vp)
    uhp.interpolate(uh)
    uDp = fem.Function(Vp)
    expr = fem.Expression(u_ex, Vp.element.interpolation_points())
    uDp.interpolate(expr)

    error_form = fem.form( ufl.inner(uDp - uhp, uDp - uhp) * ufl.dx )
    error_loc = fem.assemble_scalar(error_form)
    error = np.sqrt(MPI.COMM_WORLD.reduce(error_loc, op=MPI.SUM, root=0))

    return error



def main():

    import matplotlib.pyplot as plt

    mu_val = 1e0
    p = 1

    plt.figure()

    lambda_val = 1.0e0

    Ns = [8, 16, 32, 64, 128]
    Es = []

    for N in Ns:
        error = run_problem(N, mu_val=mu_val, lambda_val=lambda_val, p=p)
        if MPI.COMM_WORLD.rank == 0:
            print(error)
        Es.append(error)

    hs = 1 / np.array(Ns, dtype=float)

    plt.loglog(hs, Es, 'k-o', label=rf"$\lambda = {lambda_val:.1e}$")

    lambda_val = 1.0e2

    Ns = [8, 16, 32, 64, 128]
    Es = []

    for N in Ns:
        error = run_problem(N, mu_val=mu_val, lambda_val=lambda_val, p=p)
        if MPI.COMM_WORLD.rank == 0:
            print(error)
        Es.append(error)

    hs = 1 / np.array(Ns, dtype=float)

    plt.loglog(hs, Es, 'k--o', label=rf"$\lambda = {lambda_val:.1e}$")

    lambda_val = 1.0e4

    Ns = [8, 16, 32, 64, 128]
    Es = []

    for N in Ns:
        error = run_problem(N, mu_val=mu_val, lambda_val=lambda_val, p=p)
        if MPI.COMM_WORLD.rank == 0:
            print(error)
        Es.append(error)

    hs = 1 / np.array(Ns, dtype=float)

    plt.loglog(hs, Es, 'k:o', label=rf"$\lambda = {lambda_val:.1e}$")

    lambda_val = 1.0e6

    Ns = [8, 16, 32, 64, 128]
    Es = []

    for N in Ns:
        error = run_problem(N, mu_val=mu_val, lambda_val=lambda_val, p=p)
        if MPI.COMM_WORLD.rank == 0:
            print(error)
        Es.append(error)

    hs = 1 / np.array(Ns, dtype=float)

    plt.loglog(hs, Es, 'k-.o', label=rf"$\lambda = {lambda_val:.1e}$")

    plt.legend()
    plt.xlabel("$h$")
    plt.ylabel(r"$||u_h - u_\mathrm{ex}||_{L^2}$")
    plt.title(rf"$p={p}, \mu = {mu_val:.1e}$")

    plt.savefig(f"figures/locking_conv_p{p}.png", dpi=200)

    plt.show()

    


    return



if __name__ == "__main__":
    main()

