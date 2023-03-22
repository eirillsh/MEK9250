import gmsh
import numpy as np
from mpi4py import MPI

#You don't really care what happens around the cylinder for the stability analysis, so you can just create the mesh uniformly 

gmsh.initialize()
# Parameters
H = 0.41  # Channel height
D = 0.1  # Cylinder diameter
L = 2.2  # Channel length
R = 0.05  # Cylinder radius
x_dist_to_cylinder = 0.15  # Distance from left to cylinder tip
y_dist_to_cylinder = 0.15  # Distance from bottom to cylinder tip
cylinder_x, cylinder_y = 0.2, 0.2  # Coordinates of cylinder center

gdim = 2
model_rank = 0
mesh_comm = MPI.COMM_WORLD

# Creating channel and cylinder
if mesh_comm.rank == model_rank:
    channel = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    cylinder = gmsh.model.occ.addDisk(cylinder_x, cylinder_y, 0, R, R)

# Removing cylinder from fluid domain
if mesh_comm.rank == model_rank:
    fluid_domain = gmsh.model.occ.cut([(gdim, channel)], [(gdim, cylinder)])
    gmsh.model.occ.synchronize()


fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")

# Tagging different surfaces of the mesh
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(
            boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H/2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H/2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

# Refining mesh cells close to the cylinder for higher precision
refine = True
if refine:
    res_min = R / 3
    if mesh_comm.rank == model_rank:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", R)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


lf=0.5
if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", lf)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

gmsh.write("circle_in_Rectangle_LF_refined"+str(lf)+".msh")
gmsh.finalize()

#Refined mesh for implicit solver. lf = 0.0625
#Refined mesh for explicit solver. lf = 0.0625