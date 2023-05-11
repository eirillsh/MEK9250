
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import ufl


def run_problem(N):

    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)
    tdim = domain.topology.dim
    fdim = tdim - 1

    V_cg2 = ufl.VectorElement("CG", domain.ufl_cell(), 2)
    Q_cg1 = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, V_cg2)
    Q = fem.FunctionSpace(domain, Q_cg1)

    mel = ufl.MixedElement([V_cg2, Q_cg1])
    W = fem.FunctionSpace(domain, mel)

    mu = fem.Constant(domain, ScalarType(1.0))
    lambda_ = fem.Constant(domain, ScalarType(10.0))

    def u_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    
    x = ufl.SpatialCoordinate(domain)

    u_ex_1 = 16*x[0]*(1 - x[0])*x[1]*(1 - x[1])
    u_ex_2 = -0.5*u_ex_1
    u_ex = ufl.as_vector((u_ex_1, u_ex_2))
    u_ex_expr = fem.Expression(u_ex, V.element.interpolation_points())

    f = -mu * ufl.div(ufl.grad(u_ex)) - (mu+lambda_) * ufl.grad(ufl.div(u_ex))

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

    u_ex = fem.Function(V)
    u_ex.interpolate(u_bc)
    u_ex.name = "u_ex"

    return uh, ph, domain, u_ex

N = 60

uh, ph, domain, u_ex = run_problem(N)

from dolfinx import io
with io.XDMFFile(domain.comm, "output/stokes_test_output2.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(u_ex)
    xdmf.write_function(ph)
