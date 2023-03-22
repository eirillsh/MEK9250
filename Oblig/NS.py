import ufl
import os

import pyvista
import tqdm.autonotebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dolfinx as dfx

from dolfinx import fem, io, plot

from ufl import grad, inner, dx, ds, dot, nabla_grad, div
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import (XDMFFile, distribute_entity_data, gmshio)

from petsc4py.PETSc import ScalarType
import matplotlib.pyplot as plt
import pylab


# Bringing over values from previous file

# Facet markers
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5

H = 0.41  # Channel height
gdim = 2
facets_dim = 1
model_rank = 0
mesh_comm = MPI.COMM_WORLD


class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):

        # Case 1
        values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
        U_m = 0.3
        values[0] = 4 * U_m * x[1] * (H - x[1])/H**2

        return values


class Solve_2D_cylinder_NS:

    def __init__(self, mesh, facet_tags, identifier_string, scheme='e', t=0, T=8, dt=1e-3, deg_V=2):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.identifier = identifier_string
        self.scheme = scheme
        self.t = t
        self.T = T
        self.deg_V = deg_V
        self.dt = dt

    def Solve(self, plot_solution=False, create_figures=False):
        # Extract necessary things
        case = self.identifier
        mesh = self.mesh
        facet_tags = self.facet_tags
        t = self.t
        T = self.T
        dt = self.dt
        num_steps = int(T/dt)

        k = fem.Constant(mesh, PETSc.ScalarType(dt))
        mu = fem.Constant(mesh, PETSc.ScalarType(0.001))

        # Quadratic or linear Lagrange elements
        v_cg2 = ufl.VectorElement("CG", mesh.ufl_cell(), self.deg_V)
        # Linear Lagrange elements
        s_cg1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        V = fem.FunctionSpace(mesh, v_cg2)  # Velocity function space
        Q = fem.FunctionSpace(mesh, s_cg1)  # Pressure function space

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

        # the most recently computed approximation of u_{n+1}
        u_ = fem.Function(V)
        u_n = fem.Function(V)    # u_n
        u_n.name = "u_n"

        p_n = fem.Function(Q)   # p_n
        p_n.name = "p_n"
        phi = fem.Function(Q)   # pressure correcting function

        # Getting Boundary Conditions
        bcu, bcp = self.BC(V, Q)

        # Step-1 : Tentative velocity

        if self.scheme == 'e':
            print("Applying explicit solver")
            # Do I have to separate u and u_n here, or is ufl.lhs smart enough to separate for me?
            F1 = dot(u, v) * dx
            F1 -= dot(u_n, v) * dx
            F1 += k * dot(dot(u_n, nabla_grad(u_n)), v) * \
                dx          # convective term
            F1 += k * mu * inner(nabla_grad(u_n),
                                 nabla_grad(v)) * dx  # viscous term
            F1 -= k * p_n * (div(v)) * dx

            # Bilinear and Linear forms
            # independent of time, and has to be assembled once
            a1 = fem.form(ufl.lhs(F1))
            L1 = fem.form(ufl.rhs(F1))  # dependent on u_n

            A1 = fem.petsc.assemble_matrix(a1, bcs=bcu)
            b1 = fem.petsc.create_vector(L1)
            A1.assemble()
        else:
            print("Applying implicit solver")
            F1 = dot(u - u_n, v) * dx
            F1 += k * dot(dot(u_n, nabla_grad(u)), v) * dx
            F1 += k * mu * inner(grad(u), grad(v)) * dx
            F1 -= k * p_n * div(v) * dx

            # Bilinear and Linear forms
            a1 = fem.form(ufl.lhs(F1))
            L1 = fem.form(ufl.rhs(F1))

            A1 = fem.petsc.create_matrix(a1)
            b1 = fem.petsc.create_vector(L1)

        # Step-2 : Solve Poisson's equation to correct for the pressure
        a2 = fem.form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = fem.form(- div(u_)/k * q * dx)
        A2 = fem.petsc.assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = fem.petsc.create_vector(L2)

        # Step-3 : Update velocity
        a3 = fem.form(dot(u, v) * dx)
        L3 = dfx.fem.form(inner(u_, v) * dx - k * inner(grad(phi), v) * dx)
        A3 = fem.petsc.assemble_matrix(a3)
        A3.assemble()
        b3 = fem.petsc.create_vector(L3)

        # Configuring Solvers :
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)

        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)

        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)

        # Drag, lift, and pressure computations
        D = 0.1                     # Ball diameter
        U_bar = 0.2                 # Mean velocity for Re = 20 steady case
        n = -ufl.FacetNormal(mesh)  # Normal pointing out of obstacle
        dObs = ufl.Measure(
            "ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=obstacle_marker)
        u_t = inner(ufl.as_vector((n[1], -n[0])), u_)
        drag = fem.form(2 / (D * U_bar ** 2) *
                        (mu * inner(grad(u_t), n) * n[1] - p_n * n[0]) * dObs)
        lift = fem.form(-2 / (D * U_bar ** 2) *
                        (mu * inner(grad(u_t), n) * n[0] + p_n * n[1]) * dObs)
        C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
        t_array = np.linspace(dt, T, num_steps, dtype=np.float64)

        # Computing pressure difference
        tree = dfx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
        cell_candidates = dfx.geometry.compute_collisions(tree, points)
        colliding_cells = dfx.geometry.compute_colliding_cells(
            mesh, cell_candidates, points)
        front_cells = colliding_cells.links(0)
        back_cells = colliding_cells.links(1)
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

        # Solving the problem
        Explicit = not (self.scheme == 'e')
        progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
        u_filename = f"{self.identifier}_u.bp"
        p_filename = f"{self.identifier}_p.bp"

        for i in range(num_steps):
            progress.update(1)
            t += dt
            # Step 1: tentative velocity
            if Explicit:
                A1.zeroEntries()
                fem.petsc.assemble_matrix(A1, a1, bcs=bcu)
                A1.assemble()
            with b1.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b1, L1)
            fem.petsc.apply_lifting(b1, [a1], [bcu])
            fem.petsc.set_bc(b1, bcu)
            solver1.solve(b1, u_.vector)

            # Step 2: Pressure corrrection step

            with b2.localForm() as loc_2:
                loc_2.set(0)
            fem.petsc.assemble_vector(b2, L2)
            fem.petsc.apply_lifting(b2, [a2], [bcp])
            fem.petsc.set_bc(b2, bcp)

            solver2.solve(b2, phi.vector)
            p_n.x.array[:] = p_n.x.array[:] + phi.x.array[:]

            # Step 3: Velocity correction step
            with b3.localForm() as loc_3:
                loc_3.set(0)
            fem.petsc.assemble_vector(b3, L3)
            solver3.solve(b3, u_.vector)

            # if np.max(u_.x.array[:])>100:
            #    return "unstable"
            # Update solutions
            u_n.x.array[:] = u_.x.array[:]

            # COMPUTE PHYSICAL QUANTITIES
            drag_coeff = fem.assemble_scalar(drag)
            lift_coeff = fem.assemble_scalar(lift)
            p_front = None
            if len(front_cells) > 0:
                p_front = p_n.eval(points[0], front_cells[:1])

            p_back = None
            if len(back_cells) > 0:
                p_back = p_n.eval(points[1], back_cells[:1])

            C_D[i] = drag_coeff
            C_L[i] = lift_coeff
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff[i] = pressure
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff[i] -= pressure
                    break
        if plot_solution:
            pyvista.start_xvfb()
            topology, cell_types, geometry = dfx.plot.create_vtk_mesh(V)

            values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
            values[:, :len(u_n)] = u_n.x.array.real.reshape(
                (geometry.shape[0], len(u_n)))

            # Create a point cloud of glyphs
            function_grid = pyvista.UnstructuredGrid(
                topology, cell_types, geometry)
            function_grid["u"] = values
            glyphs = function_grid.glyph(orient="u", factor=0.2)

            # Create a pyvista-grid for the mesh
            grid = pyvista.UnstructuredGrid(
                *dfx.plot.create_vtk_mesh(mesh, mesh.topology.dim))

            # Create plotter
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, style="wireframe", color="k")
            plotter.add_mesh(glyphs)
            plotter.view_xy()
            if not pyvista.OFF_SCREEN:
                plotter.show()
            else:
                fig_as_array = plotter.screenshot("glyphs.png")

            p_topology, p_cell_types, p_geometry = dfx.plot.create_vtk_mesh(Q)
            u_grid = pyvista.UnstructuredGrid(
                p_topology, p_cell_types, p_geometry)
            u_grid.point_data["p"] = p_n.x.array.real
            u_grid.set_active_scalars("p")
            u_plotter = pyvista.Plotter()
            u_plotter.add_mesh(u_grid, show_edges=True)
            u_plotter.view_xy()
            if not pyvista.OFF_SCREEN:
                u_plotter.show()

        print(C_D[-1], p_diff[-1])

        if create_figures:
            # Print Physical Quantities to file
            if not os.path.exists("figures"):
                os.mkdir("figures")
            num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
            num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global

            fig = plt.figure(figsize=(25, 8))
            l1 = plt.plot(t_array, C_D, label=r"FEniCSx  ({0:d} dofs)".format(
                num_velocity_dofs+num_pressure_dofs), linewidth=2)
            plt.title("Drag coefficient")
            plt.grid()
            plt.legend()
            plt.savefig("figures/drag_comparison.png")

            fig = plt.figure(figsize=(25, 8))
            l1 = plt.plot(t_array, C_L, label=r"FEniCSx  ({0:d} dofs)".format(
                num_velocity_dofs+num_pressure_dofs), linewidth=2)
            plt.title("Lift coefficient")
            plt.grid()
            plt.legend()
            plt.savefig("figures/lift_comparison.png")

            fig = plt.figure(figsize=(25, 8))
            l1 = plt.plot(t_array, p_diff, label=r"FEniCSx ({0:d} dofs)".format(
                num_velocity_dofs+num_pressure_dofs), linewidth=2)
            plt.title("Pressure difference")
            plt.grid()
            plt.legend()
            plt.savefig("figures/pressure_comparison.png")

        return "stable"

    def Check_stability(self):
        # Extract necessary things
        case = self.identifier
        mesh = self.mesh
        facet_tags = self.facet_tags
        t = self.t
        T = self.T
        dt = self.dt
        num_steps = int(T/dt)

        k = fem.Constant(mesh, PETSc.ScalarType(dt))
        mu = fem.Constant(mesh, PETSc.ScalarType(0.001))

        # Quadratic or linear Lagrange elements
        v_cg2 = ufl.VectorElement("CG", mesh.ufl_cell(), self.deg_V)
        # Linear Lagrange elements
        s_cg1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        V = fem.FunctionSpace(mesh, v_cg2)  # Velocity function space
        Q = fem.FunctionSpace(mesh, s_cg1)  # Pressure function space

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

        # the most recently computed approximation of u_{n+1}
        u_ = fem.Function(V)
        u_n = fem.Function(V)    # u_n
        u_n.name = "u_n"

        p_n = fem.Function(Q)   # p_n
        p_n.name = "p_n"
        phi = fem.Function(Q)   # pressure correcting function

        # Getting Boundary Conditions
        bcu, bcp = self.BC(V, Q)

        # Step-1 : Tentative velocity

        if self.scheme == 'e':
            print("Applying explicit solver")
            # Do I have to separate u and u_n here, or is ufl.lhs smart enough to separate for me?
            F1 = dot(u, v) * dx
            F1 -= dot(u_n, v) * dx
            F1 += k * dot(dot(u_n, nabla_grad(u_n)), v) * \
                dx          # convective term
            F1 += k * mu * inner(nabla_grad(u_n),
                                 nabla_grad(v)) * dx  # viscous term
            F1 -= k * p_n * (div(v)) * dx

            # Bilinear and Linear forms
            # independent of time, and has to be assembled once
            a1 = fem.form(ufl.lhs(F1))
            L1 = fem.form(ufl.rhs(F1))  # dependent on u_n

            A1 = fem.petsc.assemble_matrix(a1, bcs=bcu)
            b1 = fem.petsc.create_vector(L1)
            A1.assemble()
        else:
            print("Applying implicit solver")
            F1 = dot(u - u_n, v) * dx
            F1 += k * dot(dot(u_n, nabla_grad(u)), v) * dx
            F1 += k * mu * inner(grad(u), grad(v)) * dx
            F1 -= k * p_n * div(v) * dx

            # Bilinear and Linear forms
            a1 = fem.form(ufl.lhs(F1))
            L1 = fem.form(ufl.rhs(F1))

            A1 = fem.petsc.create_matrix(a1)
            b1 = fem.petsc.create_vector(L1)

        # Step-2 : Solve Poisson's equation to correct for the pressure
        a2 = fem.form(dot(nabla_grad(p), nabla_grad(q)) * dx)
        L2 = fem.form(- div(u_)/k * q * dx)
        A2 = fem.petsc.assemble_matrix(a2, bcs=bcp)
        A2.assemble()
        b2 = fem.petsc.create_vector(L2)

        # Step-3 : Update velocity
        a3 = fem.form(dot(u, v) * dx)
        L3 = dfx.fem.form(inner(u_, v) * dx - k * inner(grad(phi), v) * dx)
        A3 = fem.petsc.assemble_matrix(a3)
        A3.assemble()
        b3 = fem.petsc.create_vector(L3)

        # Configuring Solvers :
        solver1 = PETSc.KSP().create(mesh.comm)
        solver1.setOperators(A1)

        solver2 = PETSc.KSP().create(mesh.comm)
        solver2.setOperators(A2)

        solver3 = PETSc.KSP().create(mesh.comm)
        solver3.setOperators(A3)

        # Computing pressure difference
        tree = dfx.geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
        cell_candidates = dfx.geometry.compute_collisions(tree, points)
        colliding_cells = dfx.geometry.compute_colliding_cells(
            mesh, cell_candidates, points)
        front_cells = colliding_cells.links(0)
        back_cells = colliding_cells.links(1)
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)

        # Solving the problem
        Explicit = not (self.scheme == 'e')
        initial_computations = int(np.ceil(0.1/dt))
        progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=initial_computations+np.max([initial_computations,50]))
        
        # Do the first twenty computations

        print("Beginning initial computations")
        for i in range(initial_computations):
            progress.update(1)
            t += dt
            # Step 1: tentative velocity
            if Explicit:
                A1.zeroEntries()
                fem.petsc.assemble_matrix(A1, a1, bcs=bcu)
                A1.assemble()
            with b1.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b1, L1)
            fem.petsc.apply_lifting(b1, [a1], [bcu])
            fem.petsc.set_bc(b1, bcu)
            solver1.solve(b1, u_.vector)

            # Step 2: Pressure corrrection step

            with b2.localForm() as loc_2:
                loc_2.set(0)
            fem.petsc.assemble_vector(b2, L2)
            fem.petsc.apply_lifting(b2, [a2], [bcp])
            fem.petsc.set_bc(b2, bcp)

            solver2.solve(b2, phi.vector)
            p_n.x.array[:] = p_n.x.array[:] + phi.x.array[:]

            # Step 3: Velocity correction step
            with b3.localForm() as loc_3:
                loc_3.set(0)
            fem.petsc.assemble_vector(b3, L3)
            solver3.solve(b3, u_.vector)


            # Update solutions
            u_n.x.array[:] = u_.x.array[:]

        # Compute the following computations with pressure computation
        print("Beginning testing for pressure stability")
        for i in range(np.max([initial_computations,50])):
            progress.update(1)
            t += dt
            # Step 1: tentative velocity
            if Explicit:
                A1.zeroEntries()
                fem.petsc.assemble_matrix(A1, a1, bcs=bcu)
                A1.assemble()
            with b1.localForm() as loc:
                loc.set(0)
            fem.petsc.assemble_vector(b1, L1)
            fem.petsc.apply_lifting(b1, [a1], [bcu])
            fem.petsc.set_bc(b1, bcu)
            solver1.solve(b1, u_.vector)

            # Step 2: Pressure corrrection step

            with b2.localForm() as loc_2:
                loc_2.set(0)
            fem.petsc.assemble_vector(b2, L2)
            fem.petsc.apply_lifting(b2, [a2], [bcp])
            fem.petsc.set_bc(b2, bcp)

            solver2.solve(b2, phi.vector)
            p_n.x.array[:] = p_n.x.array[:] + phi.x.array[:]

            # Step 3: Velocity correction step
            with b3.localForm() as loc_3:
                loc_3.set(0)
            fem.petsc.assemble_vector(b3, L3)
            solver3.solve(b3, u_.vector)


            # Update solutions
            u_n.x.array[:] = u_.x.array[:]

            # COMPUTE Pressure
            p_front = None
            if len(front_cells) > 0:
                p_front = p_n.eval(points[0], front_cells[:1])
            p_back = None
            if len(back_cells) > 0:
                p_back = p_n.eval(points[1], back_cells[:1])
            # Choose first pressure that is found from the different processors
            p_diff = p_front[0]-p_back[0]
            if p_diff>20 or p_diff <0: 
                return "unstable"

        print(np.max(u_n.x.array[:]))
        

        
        return "stable"

    def BC(self, V, Q):
        # Setting boundary conditions :  (OBS - remember to set Neumann through your equations)

        # Velocity boundary conditions :

        # Inlet boundary condition - parabolic:
        u_inlet = fem.Function(V)
        inlet_velocity = InletVelocity(self.t)
        u_inlet.interpolate(inlet_velocity)
        inlet_dofs = fem.locate_dofs_topological(
            V, facets_dim, self.facet_tags.find(inlet_marker))
        bcu_inlet = fem.dirichletbc(u_inlet, inlet_dofs)

        # Noslip on the Walls - noslip :
        # creating empty vector with correct dimensions of correct type
        u_noslip = np.array((0,) * self.mesh.geometry.dim,
                            dtype=PETSc.ScalarType)
        walls_dofs = fem.locate_dofs_topological(
            V, facets_dim, self.facet_tags.find(wall_marker))
        bcu_walls = fem.dirichletbc(u_noslip, walls_dofs, V)

        # The cylinder obstacle - noslip :
        obstacle_dofs = fem.locate_dofs_topological(
            V, facets_dim, self.facet_tags.find(obstacle_marker))
        bcu_obstacle = fem.dirichletbc(u_noslip, obstacle_dofs, V)

        bcu = [bcu_inlet, bcu_obstacle, bcu_walls]
        # Pressure boundary conditions :

        # Outlet boundary condition - homogenous dirichlet :
        outlet_dofs = fem.locate_dofs_topological(
            Q, facets_dim, self.facet_tags.find(outlet_marker))
        bcp_outlet = fem.dirichletbc(PETSc.ScalarType(0), outlet_dofs, Q)

        bcp = [bcp_outlet]

        return bcu, bcp
