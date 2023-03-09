import ufl

import numpy   as np
import scipy   as sp
import pyvista as pv
import dolfinx as dfx
import matplotlib.pyplot as plt

from ufl      import dx, div, grad, inner
from mpi4py   import MPI
from petsc4py import PETSc

# Marker values
LEFT   = 1
RIGHT  = 2
BOTTOM = 3
TOP    = 4

# Boundary locator functions
def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

def bottom_boundary(x):
    return np.isclose(x[1], 0.0)

def top_boundary(x):
    return np.isclose(x[1], 1.0)

# Boundary marker function
def mark_boundaries(mesh):
    facet_dim = mesh.topology.dim - 1

    # Generate mesh topology
    mesh.topology.create_entities(facet_dim)

    num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets
    facet_marker = np.full(num_facets, 0, dtype=np.int32) # Pre-allocate facet marker array

    # Left boundary
    left_boundary_facets   = dfx.mesh.locate_entities_boundary(mesh, facet_dim, left_boundary)
    facet_marker[left_boundary_facets] = LEFT

    # Right boundary
    right_boundary_facets  = dfx.mesh.locate_entities_boundary(mesh, facet_dim, right_boundary)
    facet_marker[right_boundary_facets] = RIGHT

    # Bottom boundary
    bottom_boundary_facets = dfx.mesh.locate_entities_boundary(mesh, facet_dim, bottom_boundary)
    facet_marker[bottom_boundary_facets] = BOTTOM

    # Top boundary
    top_boundary_facets    = dfx.mesh.locate_entities_boundary(mesh, facet_dim, top_boundary)
    facet_marker[top_boundary_facets] = TOP

    boundaries = dfx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype = np.int32), facet_marker)

    return boundaries

class AnalyticalSolution:
    """ Class that defines the analytical solution expression.
    """
    def __init__(self, mu):
        self.mu = mu
    
    def __call__(self, x):
        return (np.exp(x[0] / self.mu) - 1) / (np.exp(1 / self.mu) - 1)

def calc_L2_error_norm(u_h, mu, degree_raise = 3):
    """ Calculate the L2 error norm for the finite element approximation u_h. The error is interpolated into a higher-order function space.

    Parameters
    ----------
    u_h : dolfinx.fem.function.Function
        The finite element approximation of the solution u.
    mu : float
        Diffusion (viscosity) coefficient in the convection-diffusion PDE.
    degree_raise : int, optional
        The number of degrees the higher-order function space is higher than the function space of u_h, by default 3

    Returns
    -------
    L2_error_norm
        The error norm e_h = u - u_h measured in L^2(Omega).
    """
    # Create higher-order function space for interpolation of the error
    family = u_h.function_space.ufl_element().family()
    degree = u_h.function_space.ufl_element().degree()
    mesh   = u_h.function_space.mesh

    W = dfx.fem.FunctionSpace(mesh, element = (family, degree + degree_raise))

    u_h_refined = dfx.fem.Function(W)
    u_h_refined.interpolate(u_h)

    u_analytical_expr = AnalyticalSolution(mu)
    u_analytical = dfx.fem.Function(W)
    u_analytical.interpolate(u_analytical_expr)

    L2_error_norm = dfx.fem.form((u_analytical - u_h_refined) ** 2 * dx)
    L2_error_norm = dfx.fem.assemble_scalar(L2_error_norm)
    L2_error_norm = np.sqrt(L2_error_norm)
    
    return L2_error_norm

def calc_H1_error_norm(u_h, mu, degree_raise = 3):
    """ Calculate the H1 error norm for the finite element approximation u_h. The error is interpolated into a higher-order function space.

    Parameters
    ----------
    u_h : dolfinx.fem.function.Function
        The finite element approximation of the solution u.
    mu : float
        Diffusion (viscosity) coefficient in the convection-diffusion PDE.
    degree_raise : int, optional
        The number of degrees the higher-order function space is higher than the function space of u_h, by default 3

    Returns
    -------
    H1_error_norm
        The error norm e_h = u - u_h measured in H^1(Omega).
    """
    # Create higher-order function space for interpolation of the error
    family = u_h.function_space.ufl_element().family()
    degree = u_h.function_space.ufl_element().degree()
    mesh   = u_h.function_space.mesh

    W = dfx.fem.FunctionSpace(mesh, element = (family, degree + degree_raise))

    u_h_refined = dfx.fem.Function(W)
    u_h_refined.interpolate(u_h)

    u_analytical_expr = AnalyticalSolution(mu)
    u_analytical = dfx.fem.Function(W)
    u_analytical.interpolate(u_analytical_expr)

    H1_error_norm = dfx.fem.form(((u_analytical - u_h_refined) ** 2 + grad(u_analytical - u_h_refined) ** 2) * dx)
    H1_error_norm = dfx.fem.assemble_scalar(H1_error_norm)
    H1_error_norm = np.sqrt(H1_error_norm)

    return H1_error_norm

#####------CONVECTION-DIFFUSION PROBLEM------#####
L2_error_norms, H1_error_norms, hs = [], [], []
all_L2_error_norms, all_H1_error_norms = [], []
fig_idx = 1 # Figure index

SUPG = True # If True, use Streamline Upwinding Petrov-Galerkin method

# Loop over different values of mu
for mu_value in [1, 0.3, 0.1]:

    # Loop over choices for the number of grid points in x and y direction
    for N in [8, 16, 32, 64]:
        # Constants
        h = 1/N # Grid spacing
        hs.append(h)

        # Create mesh
        mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        boundaries = mark_boundaries(mesh)

        ds = ufl.Measure('ds', mesh, subdomain_data = boundaries) # Boundary integral measure
        n = ufl.FacetNormal(mesh) # Mesh normal vector 

        # Create finite elements and function spaces
        P1 = ufl.FiniteElement(family = 'CG', cell = mesh.ufl_cell(), degree = 1)
        V  = dfx.fem.FunctionSpace(mesh, P1)

        mu = dfx.fem.Constant(mesh, PETSc.ScalarType(mu_value))
        w  = ufl.as_vector((1, 0)) # Convection velocity
        f = dfx.fem.Constant(mesh, PETSc.ScalarType(0.0, ) * mesh.topology.dim) # Source term

        # Trial and test functions
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        u_h  = dfx.fem.Function(V) # Solution variable
            

        #####---------DEFINE AND SOLVE PROBLEM---------#####
        if SUPG:
            # Streamline Upwinding Petrov-Galerkin method
            supg_parameter = 0.5
            # Bilinear form
            a = dfx.fem.form((mu * inner(grad(u), grad(v)) + inner(w, grad(u)) * v 
                           + supg_parameter * h * inner(w, grad(u)) * inner(w, grad(v))) * dx)
                           #- beta * h * mu * div(grad(u)) * inner(w, grad(v))) * dx)

            # Linear form
            L = dfx.fem.form((inner(f, v) + supg_parameter * h *  inner(f * w, grad(v))) * dx)

        else:
            # Standard Galerkin
            # Bilinear form
            a = dfx.fem.form((mu * inner(grad(u), grad(v)) + inner(w, grad(u)) * v) * dx)

            # Linear form
            L = dfx.fem.form(inner(f, v) * dx)

        # Dirichlet boundary conditions
        left_BC_fun   = dfx.fem.Function(V)
        left_BC_fun.x.set(0.0) # Left boundary u = 0
        left_BC_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundaries.find(LEFT))
        left_BC = dfx.fem.dirichletbc(left_BC_fun, left_BC_dofs)

        right_BC_fun = dfx.fem.Function(V)
        right_BC_fun.x.set(1.0) # Right boundary u = 1
        right_BC_dofs = dfx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundaries.find(RIGHT))
        right_BC = dfx.fem.dirichletbc(right_BC_fun, right_BC_dofs)

        # Collect BCs
        bcs = [left_BC, right_BC]

        # Assemble system matrix
        A = dfx.fem.petsc.assemble_matrix(a, bcs = bcs)
        A.assemble()

        # Assemble right-hand side vector
        b = dfx.fem.petsc.create_vector(L)
        dfx.fem.petsc.assemble_vector(b, L)
        dfx.fem.petsc.apply_lifting(b, [a], bcs = [bcs])
        b.ghostUpdate(addv = PETSc.InsertMode.ADD_VALUES, mode = PETSc.ScatterMode.REVERSE)   
        dfx.fem.petsc.set_bc(b, bcs = bcs)

        # Create linear solver and set options
        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")

        # Calculate the solution
        solver.solve(b, u_h.vector)


        #####---------ERROR ANALYSIS--------#####
        L2_error_norm = calc_L2_error_norm(u_h, mu_value)
        H1_error_norm = calc_H1_error_norm(u_h, mu_value)

        print(f"Error analysis with mu = {mu_value}, N = {N}\n")
        print(f"L2 error norm: {L2_error_norm:.2e}")
        print(f"H1 error norm: {H1_error_norm:.2e}")

        L2_error_norms.append(L2_error_norm)
        H1_error_norms.append(H1_error_norm)

    # Error estimates
    alpha, intercept_alpha, rho_alpha, pval_alpha, stderr_alpha = sp.stats.linregress(np.log(hs), np.log(H1_error_norms))
    beta , intercept_beta , rho_beta , pval_beta , stderr_beta  = sp.stats.linregress(np.log(hs), np.log(L2_error_norms))

    C_alpha = np.exp(intercept_alpha)
    C_beta  = np.exp(intercept_beta)

    print(f"alpha = {alpha:.5f}, C_alpha = {C_alpha:.5f}")
    print(f"alpha metrics:\n Correlation coefficient = {rho_alpha:.3f}, p_value = {pval_alpha:.3f}, standard error of slope: {stderr_alpha:.3f}\n")
    print(f"beta = {beta:.5f}, C_beta = {C_beta:.5f}")
    print(f"beta metrics:\n Correlation coefficient = {rho_beta:.3f}, p_value = {pval_beta:.3f}, standard error of slope: {stderr_beta:.3f}\n")

    # Append error norms to lists of error norms for all mu values
    all_L2_error_norms.append(L2_error_norms)
    all_H1_error_norms.append(H1_error_norms)
    print("--------------------------")
    # Plot errors
    plt.figure(fig_idx)
    plt.plot(np.log(hs), np.log(L2_error_norms))
    plt.plot(np.log(hs), intercept_beta + beta * np.log(hs), '^')
    plt.title(rf"L2 Error Norm Convergence for $\mu$ = {mu_value}")
    plt.xlabel(r"$h$")

    plt.figure(fig_idx + 1)
    plt.plot(np.log(hs), np.log(H1_error_norms))
    plt.plot(np.log(hs), intercept_alpha + alpha * np.log(hs), '^')
    plt.title(rf"H1 Error Norm Convergence for $\mu$ = {mu_value}")
    plt.xlabel(r"$h$")
    fig_idx += 2

    plt.show()

    # Clear error norm lists
    L2_error_norms, H1_error_norms = [], []
    hs = []


# #####---------VISUALIZATION---------#####


# # Prepare data grid
# cells, topology, x = dfx.plot.create_vtk_mesh(V)
# grid = pv.UnstructuredGrid(cells, topology, x)
# grid.point_data["u"] = u_h.x.array.real
# grid.set_active_scalars("u")

# # Create plot window and add data
# pl = pv.Plotter()
# pl.add_mesh(grid, show_edges = True)
# pl.view_xy()
# pl.show()

# # Plot errors
# plt.figure()
# plt.loglog(L2_error_norms, hs)
# plt.title("L2 Error Norm Convergence")
# plt.xlabel(r"$h$")

# plt.figure()
# plt.loglog(H1_error_norms, hs)
# plt.title("H1 Error Norm Convergence")
# plt.xlabel(r"$h$")

# plt.show()