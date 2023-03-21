import sys

from mpi4py        import MPI 
from dolfinx.io    import gmshio
from navier_stokes import NavierStokesSolver


# Set solver parameters
scheme_type = "implicit"
initial_time = 0  # Start-time of simulation [s]
final_time = 8    # End-time of simulation [s]
velocity_element_degree = 2

if __name__ == '__main__':
    for length_factor in [0.25, 0.125, 0.0625, 0.03125, 0.015625]:
        mesh_filename = "cylinder_in_box_LF=" + str(length_factor) + ".msh"
        mesh, _, facet_tags = gmshio.read_from_msh(mesh_filename, MPI.COMM_WORLD, rank = 0, gdim = 2)

        for timestep in [1/100, 1/200, 1/400, 1/800, 1/1600]:
            if velocity_element_degree == 1:
                filename_prefix = "implicit_output/" + scheme_type + "_linear_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"
            elif velocity_element_degree == 2:
                filename_prefix = "implicit_output/" + scheme_type + "_quadratic_V_elements_dt=" + str(timestep) + "_mesh_LF=" + str(length_factor) + "_"
            else:
                raise ValueError("Choose either 1 or 2 for the velocity element degree.")

            print("#####----------------------#####")
            print(f"Running simulation with an {scheme_type} scheme and parameters: ")
            print(f"Timestep size: {timestep}")
            print(f"Mesh length factor: {length_factor}")
            print(f"Velocity element degree: {velocity_element_degree}")
            print("#####----------------------#####\n\n")

            solver = NavierStokesSolver(mesh,
                                        facet_tags,
                                        scheme = scheme_type,
                                        case_str = filename_prefix,
                                        t  = initial_time,
                                        T  = final_time,
                                        dt = timestep,
                                        degree_V = velocity_element_degree)
            solver.run_simulation()