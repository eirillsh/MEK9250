import sys 

from NS import Solve_2D_cylinder_NS
from mpi4py        import MPI 
from dolfinx.io    import gmshio

# Load mesh
lf = 1.0#sys.argv[1]
mesh_filename = "circle_in_Rectangle_LF" + str(lf) + ".msh"
mesh, cell_tags, facet_tags = gmshio.read_from_msh(r"meshes/"+mesh_filename, MPI.COMM_WORLD, rank = 0, gdim = 2)

# Set parameters
t0      = 0  
T       = 10    
dt      = 1/(2**4)
deg_V   = 2 #int(sys.argv[2]) 
scheme  = 'e' #sys.argv[3]

filename = f"{scheme}_deg_V={deg_V}_dt={dt}_mesh_lf={lf}"

if __name__ == '__main__':

    print(f"Solving using : \nLF = {lf}\ndeg_V = {deg_V}")
    solver = Solve_2D_cylinder_NS(mesh,
                                facet_tags,
                                identifier_string = filename,
                                scheme = scheme,
                                t  = t0,
                                T  = T,
                                dt = dt,
                                deg_V = deg_V)
    status = solver.Solve(create_figures=True)
    print(status)


