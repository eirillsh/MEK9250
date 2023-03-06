import ufl
import gmsh

import numpy   as np
import pyvista as pv
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl      import dx, div, grad, inner, nabla_grad
from mpi4py   import MPI
from petsc4py import PETSc

INLET, OUTLET, WALL, CYLINDER = 2, 3, 4, 5 # Facet marker values

class InletVelocity():
    """ Class that defines the function expression for the inlet velocity boundary condition.
    """
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        velocity = np.zeros((gmsh_dim, x.shape[1]), dtype = PETSc.ScalarType)
        velocity[0] = 4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
        return velocity


# Load mesh
mesh, _, facet_tags = dfx.io.gmshio.read_from_msh(MPI.COMM_WORLD, "cylinder_in_box.msh")
facet_tags.name = "Facet markers"
gmsh_dim = mesh.topology.dim
facet_dim = gmsh_dim - 1

# Physical and numerical parameters
t = 0 # Initial time
T = 1 # Final time
dt = 1/1500 # Timestep size
num_timesteps = int(T/dt) # Number of timesteps

k   = dfx.fem.Constant(mesh, PETSc.ScalarType(dt))
mu  = dfx.fem.Constant(mesh, PETSc.ScalarType(1e-3)) # Dynamic viscosity of fluid
rho = dfx.fem.Constant(mesh, PETSc.ScalarType(1))    # Fluid density

# Finite elements and function spaces
P2 = ufl.VectorElement('CG', mesh.ufl_cell(), 2) # Quadratic Lagrange elements
P1 = ufl.FiniteElement('CG', mesh.ufl_cell(), 1) # Linear Lagrange elements

V = dfx.fem.FunctionSpace(mesh, P2) # Velocity function space
Q = dfx.fem.FunctionSpace(mesh, P1) # Pressure function space

# Trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V) # Velocity
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q) # Pressure

u_ = dfx.fem.Function(V) # Velocity at previous timestep
u_.name = "u"
u_s, u_n, u_n1 = dfx.fem.Function(V), dfx.fem.Function(V), dfx.fem.Function(V) # Velocity functions for operator splitting scheme

p_ = dfx.fem.Function(Q) # Pressure at previous timestep
p_.name = "p"
phi = dfx.fem.Function(Q) # Pressure correction function


## Velocity boundary conditions
# Inlet boundary condition - prescribed velocity profile
u_inlet_expr = InletVelocity(t)
u_inlet = dfx.fem.Function(V)
u_inlet.interpolate(u_inlet_expr)
inlet_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facet_tags.find(INLET))
bc_inlet   = dfx.fem.dirichletbc(u_inlet, inlet_dofs)

# Wall boundary condition - noslip
u_noslip  = np.array((0.0, ) * gmsh_dim, dtype = PETSc.ScalarType)
wall_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facet_tags.find(WALL))
bc_wall   = dfx.fem.dirichletbc(u_noslip, wall_dofs)

# Cylinder boundary condition - noslip
cyl_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facet_tags.find(CYLINDER))
bc_cyl   = dfx.fem.dirichletbc(u_noslip, cyl_dofs)

bc_velocity = [bc_inlet, bc_wall, bc_cyl]

## Pressure boundary condition
# Outlet boundary condition - pressure equal to zero
outlet_dofs = dfx.fem.locate_dofs_topological(Q, facet_dim, facet_tags.find(OUTLET))
bc_outlet   = dfx.fem.dirichletbc(PETSc.ScalarType(0), outlet_dofs)

bc_pressure = [bc_outlet]

# Variational form - first step
f   = dfx.fem.Constant(mesh, PETSc.ScalarType((0, 0)))
F1  = rho / k * inner(u - u_n, v) * dx
F1 += inner(inner(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += inner(f, v) * dx

# Bilinear and linear form
a1 = dfx.fem.form(ufl.lhs(F1))
L1 = dfx.fem.form(ufl.rhs(F1))

# Create matrix and vector
A1 = dfx.fem.petsc.create_matrix(a1)
b1 = dfx.fem.petsc.create_vector(L1)

# Variational form - second step
a2 = dfx.fem.form(inner(grad(p), grad(q)) * dx)
L2 = dfx.fem.form(-rho / k * inner(div(u_s), q) * dx)

# Create matrix and vector
A2 = dfx.fem.petsc.assemble_matrix(a2, bcs = bc_pressure)
A2.assemble()
b2 = dfx.fem.petsc.create_vector(L2)

# Variational form - third step
a3 = dfx.fem.form(rho * inner(u, v) * dx)
L3 = dfx.fem.form((rho * inner(u_s, v) - k * inner(nabla_grad(phi), v)) * dx)

# Create matrix and vector
A3 = dfx.fem.petsc.assemble_matrix(a3)
A3.assemble()
b3 = dfx.fem.petsc.create_vector(L3)

## Configure solvers for the iterative solution steps
# Step 1
solver1 = PETSc.KSP().create(mesh.comm)
solver1.setOperators(A1)
solver1.setType(PETSc.KSP.Type.BCGS)
pc1 = solver1.getPC()
pc1.setType(PETSc.PC.Type.JACOBI)

# Step 2
solver2 = PETSc.KSP().create(mesh.comm)
solver2.setOperators(A2)
solver2.setType(PETSc.KSP.Type.MINRES)
pc2 = solver2.getPC()
pc2.setType(PETSc.PC.Type.HYPRE)
pc2.setHYPREType("boomeramg")

# Step 3
solver3 = PETSc.KSP().create(mesh.comm)
solver3.setOperators(A3)
solver3.setType(PETSc.KSP.Type.CG)
pc3 = solver3.getPC()
pc3.setType(PETSc.PC.Type.SOR)