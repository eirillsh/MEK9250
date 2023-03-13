import sys

from mpi4py        import MPI 
from dolfinx.io    import gmshio
from navier_stokes import NavierStokesSolver

# Load mesh
length_factor = 0.0625
mesh_filename = "cylinder_in_box_LF=" + str(length_factor) + ".msh"
mesh, _, facet_tags = gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, rank = 0, gdim = 2)

# from dolfinx.io import XDMFFile
# with XDMFFile(MPI.COMM_WORLD, "mesh_file.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
# stop

# Set solver parameters
initial_time = 0  # Start-time of simulation [s]
final_time = 5    # End-time of simulation [s]
timestep = 1/1600 # Timestep size [s]
velocity_element_degree = int(sys.argv[1]) # Degree of velocity finite element function space
if sys.argv[2] == 'e':
    scheme_type = "explicit"
elif sys.argv[2] == 'i':
    scheme_type = "implicit"
else:
    raise ValueError("Choose either 'e' for explicit or 'i' for implicit scheme.")
if velocity_element_degree == 1:
    filename_prefix = scheme_type + "_linear_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"
elif velocity_element_degree == 2:
    filename_prefix = scheme_type + "_quadratic_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"
else:
    raise ValueError("Choose either 1 or 2 for the velocity element degree.")

if __name__ == '__main__':
    solver = NavierStokesSolver(mesh,
                                facet_tags,
                                scheme = scheme_type,
                                case_str = filename_prefix,
                                t  = initial_time,
                                T  = final_time,
                                dt = timestep,
                                degree_V = velocity_element_degree)
    solver.run_simulation()