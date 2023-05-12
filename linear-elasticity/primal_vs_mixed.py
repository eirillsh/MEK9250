import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, io


def compute_error(uh: fem.Function):

    Vp = fem.VectorFunctionSpace(uh.function_space.mesh, ("CG", 4))

    u_ex_expr = fem.Expression(u_ex, Vp.element.interpolation_points())
    u_ex_p = fem.Function(Vp)
    u_ex_p.interpolate(u_ex_expr)

    uh_p = fem.Function(Vp)
    uh_p.interpolate(uh)

    error_form = fem.form( ufl.inner(u_ex_p - uh_p, u_ex_p - uh_p) * ufl.dx )
    error_loc = fem.assemble_scalar(error_form)
    error = np.sqrt(MPI.COMM_WORLD.reduce(error_loc, op=MPI.SUM, root=0))

    return error

N = 60
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                 cell_type=mesh.CellType.triangle)


mu = fem.Constant(domain, ScalarType(1.0))
lambda_ = fem.Constant(domain, ScalarType(1.0e6))


x = ufl.SpatialCoordinate(domain)

phi = ufl.sin(ufl.pi * x[0] * x[1])
u_ex = ufl.as_vector((phi.dx(1), -phi.dx(0)))

lapl_u_ex_1 =  phi.dx(1).dx(0).dx(0) + phi.dx(1).dx(1).dx(1)
lapl_u_ex_2 = -phi.dx(0).dx(0).dx(0) - phi.dx(0).dx(1).dx(1)

f = -mu * ufl.as_vector((lapl_u_ex_1, lapl_u_ex_2))


def primal_solve(V: fem.FunctionSpace):

    """ Boundary conditions """
    uD_expr = fem.Expression(u_ex, V.element.interpolation_points())
    uD = fem.Function(V)
    uD.interpolate(uD_expr)

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
        (mu + lambda_) * ufl.div(u) * ufl.div(v) * ufl.dx
    l = ufl.inner(f, v) * ufl.dx

    problem = fem.petsc.LinearProblem(a, l, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    uh = problem.solve()
    uh.name = "u_h"

    return uh

def mixed_solve(V: fem.FunctionSpace, Q: fem.FunctionSpace):

    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1

    V_el = V.ufl_element()
    Q_el = Q.ufl_element()

    mel = ufl.MixedElement([V_el, Q_el])
    W = fem.FunctionSpace(domain, mel)

    def u_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    
    u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points())

    u_bc = fem.Function(V)
    u_bc.interpolate(u_ex_expr)
    u_facets = mesh.locate_entities_boundary(domain, fdim, u_boundary)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, u_facets)
    bc0 = fem.dirichletbc(u_bc, u_dofs, W.sub(0))

    bcs = [bc0]
 
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = fem.form(
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
        p * ufl.div(v) * ufl.dx + \
        (mu + lambda_) * ufl.div(u) * q * ufl.dx + \
        -p * q * ufl.dx
    )
    L = fem.form( ufl.inner(f, v) * ufl.dx )

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    fem.petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    U = fem.Function(W)
    try:
        ksp.solve(b, U.vector)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e

    uh, ph = U.sub(0).collapse(), U.sub(1).collapse()
    uh.name = "u_h"
    ph.name = "p_h"

    return uh, ph

mu.value = 1.0
lambda_.value = 1e6


V = fem.VectorFunctionSpace(domain, ("CG", 2))
# V = fem.VectorFunctionSpace(domain, ("BDM", 2))
# W = fem.FunctionSpace(domain, "BDM", 3)
# print(W)
uh_primal = primal_solve(V)
eh_primal = compute_error(uh_primal)

V = fem.VectorFunctionSpace(domain, ("CG", 2))
Q = fem.FunctionSpace(domain, ("CG", 1))
uh_mixed, _ = mixed_solve(V, Q)
eh_mixed = compute_error(uh_mixed)

print(eh_primal)
print(eh_mixed)

