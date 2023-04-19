# mpirun -np 4 -genv OMP_NUM_THREADS=1 python3 locking.py

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, io

N = 60
""" Edges per side """
p = 1
""" Element polynomial order """

domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                 cell_type=mesh.CellType.triangle)

# domain = mesh.create_rectangle(MPI.COMM_WORLD, [[-1,-1], [1,1]], [N, N], 
#                                  cell_type=mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", p))

mu = fem.Constant(domain, ScalarType(1.0))
lambda_ = fem.Constant(domain, ScalarType(1.0e4))

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

e = fem.Function(V)
e.vector.array[:] = uD.vector.array[:] - uh.vector.array[:]
e.name = "error"


order_increase = 2
Vp = fem.VectorFunctionSpace(domain, ("CG", p+order_increase))
uhp = fem.Function(Vp)
uhp.interpolate(uh)
uDp = fem.Function(Vp)
expr = fem.Expression(u_ex, Vp.element.interpolation_points())
uDp.interpolate(expr)


with io.XDMFFile(domain.comm, "output/locking.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(uD)
    xdmf.write_function(e)

norm_form = fem.form( ufl.inner(u_ex, u_ex) * ufl.dx)
norm_loc = fem.assemble_scalar(norm_form)
norm = np.sqrt(MPI.COMM_WORLD.reduce(norm_loc, op=MPI.SUM, root=0))

error_form = fem.form( ufl.inner(uDp - uhp, uDp - uhp) * ufl.dx )
error_loc = fem.assemble_scalar(error_form)
error = np.sqrt(MPI.COMM_WORLD.reduce(error_loc, op=MPI.SUM, root=0))

div_norm_form = fem.form( ufl.div(u_ex) * ufl.div(u_ex) * ufl.dx)
div_norm_loc = fem.assemble_scalar(div_norm_form)
div_norm = np.sqrt(MPI.COMM_WORLD.reduce(div_norm_loc, op=MPI.SUM, root=0))

f_norm_form = fem.form( ufl.inner(f, f) * ufl.dx)
f_norm_loc = fem.assemble_scalar(f_norm_form)
f_norm = np.sqrt(MPI.COMM_WORLD.reduce(f_norm_loc, op=MPI.SUM, root=0))

if MPI.COMM_WORLD.rank == 0:
    print(norm)
    print(error)
    print(div_norm)
    # print(f_norm)



