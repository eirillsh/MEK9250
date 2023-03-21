import ufl

import numpy   as np
import pandas  as pd
import dolfinx as dfx

from ufl        import dx, dot, div, grad, inner, nabla_grad
from petsc4py   import PETSc

INLET, OUTLET, WALL, CYLINDER = 2, 3, 4, 5 # Facet marker values
H = 0.41  # Height of domain [m]
U_m = 0.3 # Mean velocity [m]
U_bar = U_m * 2 / 3
gmsh_dim  = 2
facet_dim = gmsh_dim - 1
tol = 1e-6 # Tolerance for checking if steady state is achieved
write_file = False # Write results to VTX file if True

class InletVelocity():
    """ Class that defines the function expression for the inlet velocity boundary condition.
    """
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        velocity = np.zeros((gmsh_dim, x.shape[1]), dtype = PETSc.ScalarType)
        velocity[0] = 4 * U_m * x[1] * (H - x[1]) / (H**2)
        return velocity

class NavierStokesSolver():
    def __init__(self, mesh, facet_tags, case_str, scheme = 'explicit', t = 0, T = 8, dt = 1/1000, degree_V = 2):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.case_str = case_str
        self.scheme = scheme
        self.t = t
        self.T = T
        self.dt = dt
        self.degree_V = degree_V

    def run_simulation(self):

        # Get filename string
        case_str = self.case_str

        # Extract mesh
        mesh       = self.mesh
        facet_tags = self.facet_tags

        # Physical and numerical parameters
        t = self.t # Initial time
        T = self.T # Final time
        dt = self.dt # Timestep size
        num_timesteps = int(T/dt) # Number of timesteps
        t_0 = None
        i_ss = None
        steady_state_reached = False

        k   = dfx.fem.Constant(mesh, PETSc.ScalarType(dt))   # Dolfinx constant object for the timestep
        mu  = dfx.fem.Constant(mesh, PETSc.ScalarType(1e-3)) # Dynamic viscosity of fluid
        rho = dfx.fem.Constant(mesh, PETSc.ScalarType(1))    # Fluid density

        ### Finite elements and function spaces

        # Velocity: Linear Lagrange elements if self.degree_V == 1 and
        # quadratic Lagrange elements if self.degree_V == 2
        PV = ufl.VectorElement('CG', mesh.ufl_cell(), degree = self.degree_V)

        # Pressure: Linear Lagrange elements
        PP = ufl.FiniteElement('CG', mesh.ufl_cell(), 1) 

        V = dfx.fem.FunctionSpace(mesh, PV) # Velocity function space
        Q = dfx.fem.FunctionSpace(mesh, PP) # Pressure function space

        # Trial and test functions
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V) # Velocity
        p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q) # Pressure

        u_ = dfx.fem.Function(V) # Velocity at previous timestep
        u_.name = "u"
        u_s, u_n = dfx.fem.Function(V), dfx.fem.Function(V) # Velocity functions for operator splitting scheme

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

        # Step 1 - compute tentative velocity
        if self.scheme == 'explicit':
            ##### Explicit Euler + IPCS #####
            # Variational form
            F1  = rho / k * dot(u - u_n, v) * dx # Temporal partial derivative
            F1 += rho * inner(dot(u_n, nabla_grad(u_n)), v) * dx # Advective term
            F1 += mu * inner(grad(u_n), grad(v)) * dx # Viscous term
            F1 -= dot(p_, div(v)) * dx # Pressure gradient term
        else:
            ##### Explicit Euler + semi-implicit IPCS #####
            F1  = inner(u - u_n, v) * dx
            F1 += k * inner(dot(u_n, nabla_grad(u)), v) * dx
            F1 += k / rho * mu * inner(grad(u), grad(v)) * dx
            F1 -= k / rho * dot(p_, div(v)) * dx
        
        # Bilinear and linear form
        a1 = dfx.fem.form(ufl.lhs(F1))
        L1 = dfx.fem.form(ufl.rhs(F1))

        # Create matrix and vector
        A1 = dfx.fem.petsc.create_matrix(a1)
        b1 = dfx.fem.petsc.create_vector(L1)

        # Step 2 - solve Poission problem for the pressure correction phi
        a2 = dfx.fem.form(inner(grad(p), grad(q)) * dx)
        L2 = dfx.fem.form(-rho / k * div(u_s) * q * dx)

        # Create matrix and vector
        A2 = dfx.fem.petsc.assemble_matrix(a2, bcs = bc_pressure)
        A2.assemble()
        b2 = dfx.fem.petsc.create_vector(L2)

        # Step 3 - Update velocity
        a3 = dfx.fem.form(inner(u, v) * dx)
        L3 = dfx.fem.form(inner(u_s, v) * dx - k / rho * inner(grad(phi), v) * dx)

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
        drag = dfx.fem.form( 2 / (0.1 * U_bar ** 2) * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dS)
        lift = dfx.fem.form(-2 / (0.1 * U_bar ** 2) * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dS)

        if mesh.comm.rank == 0:
            # Pre-allocate arrays for drag and lift coefficients
            C_D = np.zeros(num_timesteps, dtype = PETSc.ScalarType)
            C_L = np.zeros(num_timesteps, dtype = PETSc.ScalarType)
            t_u = np.zeros(num_timesteps, dtype = np.float64)
            t_p = np.zeros(num_timesteps, dtype = np.float64)

            C_L_max_found = False # Boolean used to stop looking for initial time t_0

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
        if write_file:
            vtx_u = dfx.io.VTXWriter(mesh.comm, u_filename, [u_])
            vtx_p = dfx.io.VTXWriter(mesh.comm, p_filename, [p_])

        # Write initial condition to files
        if write_file:
            vtx_u.write(t)
            vtx_p.write(t)

        #####--------SOLUTION TIMELOOP--------#####
        if mesh.comm.rank == 0:
            print(f"Solving the problem with an {self.scheme} scheme.")
            print(f"Timestep size dt = {dt}")
            hmin = dfx.cpp.mesh.h(mesh, 2,
                    np.arange(mesh.topology.index_map(2).size_local + mesh.topology.index_map(2).num_ghosts, dtype = np.int32)).min()
            print(f"Smallest edge length in mesh: {hmin}")

        for i in range(num_timesteps):

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
            if write_file:
                vtx_u.write(t)
                vtx_p.write(t)

            # Update variable previous timestep variables with solution from this timestep
            with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n:#, u_n1.vector.localForm() as loc_n1:
                #loc_n.copy(loc_n1)
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

                if i > 20 and not steady_state_reached and (np.abs(C_D[i] - C_D[i-1])/C_D[i] < tol):
                    print(C_D[i])
                    print(C_D[i-1])
                    print(f"Steady state reached at t = {t}")
                    steady_state_reached = True
                    i_ss = i

                    if np.abs(C_D[i]) > 1e5:
                        print("Stopping simulation because C_D approaches inf.")
                        break
                    

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
        if write_file:
            vtx_u.close()
            vtx_p.close()


        # Print physical quantities
        if mesh.comm.rank == 0:
            i_t_0 = 0
            if i_ss is None:
                i_ss = 0
            print(f"Maximum drag coefficient: {np.max(C_D[i_ss:])}")
            print(f"Maximum lift coefficient: {np.max(C_L[i_ss:])}")
            print(f"Maximum Pressure difference: {np.max(delta_P[i_ss:])}")
            print(f"Infinity norm velocity: {np.max(u_.vector.norm(3))}")
            num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
            num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global
            print(f"Drag coefficient at t = {T}: {C_D[-1]}")
            print(f"Lift coefficient at t = {T}: {C_L[-1]}")
            print(f"Pressure difference at t = {T}: {delta_P[-1]}")

            print(f"# of velocity DOFS: {num_velocity_dofs}")
            print(f"# of pressure DOFS: {num_pressure_dofs}")

            print("#####----------------------#####\n")

        # Store the data in a pandas.DataFrame
        if mesh.comm.rank == 0:
            df = pd.DataFrame(np.array([C_D, C_L, delta_P])).T
            df = df[i_t_0:] # Cut off data prior to initial time t_0

            # Clean DataFrame and save to pickle
            df.set_index(df.index * dt, inplace = True)
            df.rename(columns = {0:'C_D', 1:'C_L', 2:'delta_P'}, inplace = True)
            df.to_pickle(case_str + "data_pickle.pkl")

            df2 = pd.DataFrame(np.array([C_D, C_L, delta_P])).T
            df2 = df2[i_ss:] # Cut off data prior to initial time t_0

            # Clean DataFrame and save to pickle
            df2.set_index(df2.index * dt, inplace = True)
            df2.rename(columns = {0:'C_D', 1:'C_L', 2:'delta_P'}, inplace = True)
            df2.to_pickle(case_str + "steady_state_data_pickle.pkl")
