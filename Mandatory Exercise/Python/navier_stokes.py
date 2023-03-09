import ufl

import numpy   as np
import pandas  as pd
import pyvista as pv
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl        import dx, dot, div, grad, inner, nabla_grad
from mpi4py     import MPI
from petsc4py   import PETSc
from dolfinx.io import gmshio

INLET, OUTLET, WALL, CYLINDER = 2, 3, 4, 5 # Facet marker values
H = 0.41  # Height of domain [m]
U_m = 1.5 # Mean velocity [m]
sinusoidal = False
if sinusoidal:
    case_str = "case_2D-3_"
else:
    case_str = "case_2D-2_"
C_L_max_found = False

class InletVelocity():
    """ Class that defines the function expression for the inlet velocity boundary condition.
    """
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        velocity = np.zeros((gmsh_dim, x.shape[1]), dtype = PETSc.ScalarType)
        if sinusoidal:
            velocity[0] = 4 * U_m * np.sin(self.t * np.pi / 8) * x[1] * (H - x[1]) / (H**2)
        else:
            velocity[0] = 4 * U_m * x[1] * (H - x[1]) / (H**2)
        return velocity

# Load mesh
gmsh_dim  = 2 # Dimension of mesh
facet_dim = gmsh_dim - 1 # Facet dimension
mesh, _, facet_tags = gmshio.read_from_msh("cylinder_in_box.msh", MPI.COMM_WORLD, rank = 0, gdim = gmsh_dim)
facet_tags.name = "Facet markers"



# Physical and numerical parameters
t = 0 # Initial time
T = 10 # Final time
dt = 1/1000 # Timestep size
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
u_noslip  = np.array((0.0, ) * mesh.geometry.dim, dtype = PETSc.ScalarType)
wall_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facet_tags.find(WALL))
bc_wall   = dfx.fem.dirichletbc(u_noslip, wall_dofs, V)

# Cylinder boundary condition - noslip
cyl_dofs = dfx.fem.locate_dofs_topological(V, facet_dim, facet_tags.find(CYLINDER))
bc_cyl   = dfx.fem.dirichletbc(u_noslip, cyl_dofs, V)

bc_velocity = [bc_inlet, bc_wall, bc_cyl]

## Pressure boundary condition
# Outlet boundary condition - pressure equal to zero
outlet_dofs = dfx.fem.locate_dofs_topological(Q, facet_dim, facet_tags.find(OUTLET))
bc_outlet   = dfx.fem.dirichletbc(PETSc.ScalarType(0), outlet_dofs, Q)

bc_pressure = [bc_outlet]

# Variational form - first step
f   = dfx.fem.Constant(mesh, PETSc.ScalarType((0, 0)))
F1  = rho / k * dot(u - u_n, v) * dx
F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
F1 += (0.5 * mu * inner(grad(u + u_n), grad(v)) - dot(p_, div(v))) * dx
F1 += dot(f, v) * dx

# Bilinear and linear form
a1 = dfx.fem.form(ufl.lhs(F1))
L1 = dfx.fem.form(ufl.rhs(F1))

# Create matrix and vector
A1 = dfx.fem.petsc.create_matrix(a1)
b1 = dfx.fem.petsc.create_vector(L1)

# Variational form - second step
a2 = dfx.fem.form(dot(grad(p), grad(q)) * dx)
L2 = dfx.fem.form(-rho / k * dot(div(u_s), q) * dx)

# Create matrix and vector
A2 = dfx.fem.petsc.assemble_matrix(a2, bcs = bc_pressure)
A2.assemble()
b2 = dfx.fem.petsc.create_vector(L2)

# Variational form - third step
a3 = dfx.fem.form( rho * dot(u, v) * dx)
L3 = dfx.fem.form((rho * dot(u_s, v) - k * dot(nabla_grad(phi), v)) * dx)

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


#####-----DRAG, LIFT and PRESSURE DIFFERENCE COMPUTATION-----#####
n  = -ufl.FacetNormal(mesh) # Normal vector pointing out of mesh
dS = ufl.Measure("ds", domain = mesh, subdomain_data = facet_tags, subdomain_id = CYLINDER) # Cylinder surface integral measure
u_t = inner(ufl.as_vector((n[1], -n[0])), u_) # Tangential velocity
drag = dfx.fem.form( 2 / (0.1) * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dS)
lift = dfx.fem.form(-2 / (0.1) * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dS)

if mesh.comm.rank == 0:
    # Pre-allocate arrays for drag and lift coefficients
    C_D = np.zeros(num_timesteps, dtype = PETSc.ScalarType)
    C_L = np.zeros(num_timesteps, dtype = PETSc.ScalarType)
    t_u = np.zeros(num_timesteps, dtype = np.float64)
    t_p = np.zeros(num_timesteps, dtype = np.float64)

# Prepare evaluation of the pressure in the front and in the back of the cylinder
tree = dfx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
cell_candidates = dfx.geometry.compute_collisions(tree, points)
colliding_cells = dfx.geometry.compute_colliding_cells(mesh, cell_candidates, points)
front_cells = colliding_cells.links(0)
back_cells  = colliding_cells.links(1)
if mesh.comm.rank == 0:
    delta_P = np.zeros(num_timesteps, dtype = PETSc.ScalarType)


# Prepare output files
u_filename = case_str + "velocity.bp"
p_filename = case_str + "pressure.bp"
vtx_u = dfx.io.VTXWriter(mesh.comm, u_filename, [u_])
vtx_p = dfx.io.VTXWriter(mesh.comm, p_filename, [p_])

# Write initial condition to files
vtx_u.write(t)
vtx_p.write(t)

# Track solver progress
# progress = tqdm.autonotebook.tqdm(desc = "Solving PDE", total = num_timesteps)

#####--------SOLUTION TIMELOOP--------#####
for i in range(num_timesteps):

    #progress.update(1)

    t += dt # Increment timestep

    # Update inlet velocity
    u_inlet_expr.t = t
    u_inlet.interpolate(u_inlet_expr)

    ## Step 1 - Tentative velocity step
    # Set BCs and assemble system matrix and vector
    A1.zeroEntries()
    dfx.fem.petsc.assemble_matrix(A1, a1, bcs = bc_velocity)
    A1.assemble()
    with b1.localForm() as loc:
        loc.set(0)
    dfx.fem.petsc.assemble_vector(b1, L1)
    dfx.fem.petsc.apply_lifting(b1, [a1], [bc_velocity])
    b1.ghostUpdate(addv = PETSc.InsertMode.ADD_VALUES, mode = PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b1, bc_velocity)

    solver1.solve(b1, u_s.vector) # Solve for the tentative velocity
    u_s.x.scatter_forward() # Collect solution from dofs computed in parallel 

    # Step 2 - Pressure correction
    with b2.localForm() as loc:
        loc.set(0)
    dfx.fem.petsc.assemble_vector(b2, L2)
    dfx.fem.petsc.apply_lifting(b2, [a2], [bc_pressure])
    b2.ghostUpdate(addv = PETSc.InsertMode.ADD_VALUES, mode = PETSc.ScatterMode.REVERSE)
    dfx.fem.set_bc(b2, bc_pressure)

    solver2.solve(b2, phi.vector) # Solve for the pressure correction function
    phi.x.scatter_forward() # Collect solution from dofs computed in parallel

    p_.vector.axpy(1, phi.vector) # Compute pressure by adding the pressure correction function phi
    p_.x.scatter_forward() 

    # Step 3 - Velocity correction
    with b3.localForm() as loc:
        loc.set(0)
    dfx.fem.petsc.assemble_vector(b3, L3)
    b3.ghostUpdate(addv = PETSc.InsertMode.ADD_VALUES, mode = PETSc.ScatterMode.REVERSE)

    solver3.solve(b3, u_.vector) # Solve for the velocity
    u_.x.scatter_forward() # Collect solution from dofs computed in parallel

    # Write solutions to files
    vtx_u.write(t)
    vtx_p.write(t)

    # Update variable previous timestep variables with solution from this timestep
    with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
        loc_n.copy(loc_n1)
        loc_.copy(loc_n)

    # Compute physical quantities
    drag_coeff = mesh.comm.gather(dfx.fem.assemble_scalar(drag), root = 0)
    lift_coeff = mesh.comm.gather(dfx.fem.assemble_scalar(lift), root = 0)
    p_front    = None
    p_back     = None

    if len(front_cells) > 0:
        p_front = p_.eval(points[0], front_cells[:1])
    p_front = mesh.comm.gather(p_front, root = 0)
    
    if len(back_cells) > 0:
        p_back = p_.eval(points[1], back_cells[:1])
    p_back = mesh.comm.gather(p_back, root = 0)
    
    if mesh.comm.rank == 0:
        t_u[i] = t
        t_p[i] = t - dt / 2
        C_D[i] = sum(drag_coeff)
        C_L[i] = sum(lift_coeff)

        if i > 50 and not C_L_max_found:
            if C_L[i] < C_L[i-1]:
                C_L_max_found = True
                t_0 = t - dt
                i_t_0 = i-1
                C_L_max = C_L[i_t_0]

        # Choose the pressure that is found first from the different processors
        for pressure in p_front:
            if pressure is not None:
                delta_P[i] = pressure[0]
                break
        for pressure in p_back:
            if pressure is not None:
                delta_P[i] -= pressure[0]
                break

# Close output files
vtx_u.close()
vtx_p.close()


# Print physical quantities
if mesh.comm.rank == 0:
    print(f"t_0 = {t_0} at timestep i = {i_t_0}")
    print(f"Maximum drag coefficient: {np.max(C_D[i_t_0:])}")
    print(f"Maximum lift coefficient: {np.max(C_L[i_t_0:])}")
    print(f"Maximum Pressure difference: {np.max(delta_P[i_t_0:])}")
    num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global
    print(f"Pressure difference at t = {T}: {delta_P[-1]}")

    print(f"# of velocity DOFS: {num_velocity_dofs}")
    print(f"# of pressure DOFS: {num_pressure_dofs}")

# Store the data in a pandas.DataFrame
if mesh.comm.rank == 0:
    df = pd.DataFrame(np.array([C_D, C_L, delta_P])).T
    df = df[i_t_0:] # Cut off data prior to initial time t_0
    
    # Clean DataFrame and save to pickle
    df.set_index(df.index * dt, inplace = True)
    df.rename(columns = {0:'C_D', 1:'C_L', 2:'delta_P'}, inplace = True)
    df.to_pickle(case_str + "data_pickle.pkl")
