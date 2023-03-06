import gmsh

import numpy   as np

from   mpi4py  import MPI

gmsh.initialize()

# Mesh and domain parameters
L = 2.2 # Length of domain [m]
H = 0.41 # Height of domain [m]
r = 0.05 # Radius of cylinder [m]
x_c = 0.15 # Distance from inlet boundary to cylinder tip in x direction [m]
y_c = 0.15 # Distance from lower boundary to cylinder bottom in y direction [m]

c_x = 0.2 # x coordinate of cylinder center [m]
c_y = 0.2 # y coordinate of cylinder center [m]

gmsh_dim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    box_domain = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag = 1)
    cylinder   = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

# Remove the cylinder from the box domain
if mesh_comm.rank == model_rank:
    fluid_domain = gmsh.model.occ.cut([(gmsh_dim, box_domain)], [(gmsh_dim, cylinder)])
    gmsh.model.occ.synchronize()

# Add physical volume marker for the fluid mesh
fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim = gmsh_dim)
    assert(len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

# Tag the boundaries of the mesh by computing the centers of mass of the boundaries
INLET, OUTLET, WALL, CYLINDER = 2, 3, 4, 5 # Marker values
inflow, outflow, walls, cyl = [], [], [], []

if mesh_comm.rank == model_rank:
    # Get boundaries and loop over all
    boundaries = gmsh.model.getBoundary(volumes, oriented = False)
    for boundary in boundaries:
        com = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(com, [0, H/2, 0]):
            # Boundary is inlet
            inflow.append(boundary[1])
        if np.allclose(com, [L, H/2, 0]):
            # Boundary is outlet
            outflow.append(boundary[1])
        if np.allclose(com, [L/2, H, 0]) or np.allclose(com, [L/2, 0, 0]):
            # Boundary is upper or lower wall
            walls.append(boundary[1])
        else:
            # Boundary is on the cylinder
            cyl.append(boundary[1])
    # Add boundary tags to the model
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, inflow, INLET)
    gmsh.model.setPhysicalName(gmsh_dim - 1, INLET, "Inlet")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, outflow, OUTLET)
    gmsh.model.setPhysicalName(gmsh_dim - 1, OUTLET, "Outlet")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, walls, WALL)
    gmsh.model.setPhysicalName(gmsh_dim - 1, WALL, "Walls")
    gmsh.model.addPhysicalGroup(gmsh_dim - 1, cyl, CYLINDER)
    gmsh.model.setPhysicalName(gmsh_dim - 1, CYLINDER, "Cylinder")

# Refine mesh cells close to the cylinder
res_min = r / 3
if mesh_comm.rank == model_rank:
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", cyl)

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

# Generate the mesh
if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gmsh_dim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

gmsh.write("cylinder_in_box.msh")