# mpirun -np 4 -genv OMP_NUM_THREADS=1 python3 locking.py

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, io

N = 10
""" Edges per side """

domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                 cell_type=mesh.CellType.triangle)

p = 2
V = fem.VectorFunctionSpace(domain, ("CG", p))
Q = fem.FunctionSpace(domain, ("CG", p-1))
# V = fem.VectorFunctionSpace(domain, ("CG", 2))
# Q = fem.FunctionSpace(domain, ("DG", 0))
W = fem.FunctionSpace(domain, ufl.MixedElement(
    V.ufl_element(), Q.ufl_element()
))

mu = fem.Constant(domain, ScalarType(1.0))
lambda_ = fem.Constant(domain, ScalarType(1.0e2))

x = ufl.SpatialCoordinate(domain)

# phi = ufl.sin(ufl.pi * x[0] * x[1])
# u_ex = ufl.as_vector((phi.dx(1), -phi.dx(0)))

# lapl_u_ex_1 =  phi.dx(1).dx(0).dx(0) + phi.dx(1).dx(1).dx(1)
# lapl_u_ex_2 = -phi.dx(0).dx(0).dx(0) - phi.dx(0).dx(1).dx(1)

# f = -mu * ufl.as_vector((lapl_u_ex_1, lapl_u_ex_2))

u_ex_1 = 16*x[0]*(1 - x[0])*x[1]*(1 - x[1])
u_ex_2 = -0.5*u_ex_1
u_ex = ufl.as_vector((u_ex_1, u_ex_2))

f = -mu * ufl.div(ufl.grad(u_ex)) - (mu+lambda_) * ufl.grad(ufl.div(u_ex))


f_expr = fem.Expression(f, W.sub(0).element.interpolation_points())
f_func = fem.Function(W.sub(0).collapse()[0])
f_func.interpolate(f_expr)
f_func.name = "f"

"""Neumann"""
# ds = ufl.Measure("ds", domain=domain)

""" Boundary conditions """

u_ex_expr = fem.Expression(u_ex, W.sub(0).element.interpolation_points())
uD = fem.Function(W.sub(0).collapse()[0])
uD.interpolate(u_ex_expr)

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)


# ml_inv = fem.Constant(domain, 1 / (mu.value + lambda_.value))

a_tot = fem.form(   mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
                    p * ufl.div(v) * ufl.dx + \
                    (mu+lambda_) * ufl.div(u) * q * ufl.dx + \
                    -p * q * ufl.dx
                )
# a_tot = fem.form(   mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + \
#                     -ufl.inner(ufl.grad(p), v) * ufl.dx + \
#                     ufl.div(u) * q * ufl.dx + \
#                     -ml_inv * p * q * ufl.dx
#                 )


l = fem.form( ufl.inner(f, v) * ufl.dx )

A = fem.petsc.assemble_matrix(a_tot, bcs=[bc])
A.assemble()

F = fem.petsc.assemble_vector(l)
fem.petsc.apply_lifting(F, [a_tot], bcs=[[bc]])
F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

fem.petsc.set_bc(F, [bc])

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

w = fem.Function(W)
try:
    ksp.solve(F, w.vector)
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

uh, ph = w.sub(0).collapse(), w.sub(1).collapse()
uh.name = "u_h"
ph.name = "p_h"


uD.name = "u_exact"



# e = fem.Function(V)
# e.interpolate(uh)
# # e.vector.array[:] = uD.vector.array[:] - uh.vector.array[:]
# e.vector.array[:] -= uD.vector.array[:]
# e.vector.array[:] *= -1
# e.name = "error"


# order_increase = 2
# Vp = fem.VectorFunctionSpace(domain, ("CG", p+order_increase))
# uhp = fem.Function(Vp)
# uhp.interpolate(uh)
# uDp = fem.Function(Vp)
# expr = fem.Expression(u_ex, Vp.element.interpolation_points())
# uDp.interpolate(expr)


with io.XDMFFile(domain.comm, "output/solid_pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(uD)
    xdmf.write_function(ph)
    xdmf.write_function(f_func)
    # xdmf.write_function(e)

