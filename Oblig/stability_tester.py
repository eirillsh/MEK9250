import sys 
import numpy as np
import matplotlib.pyplot as plt

from NS import Solve_2D_cylinder_NS
from mpi4py        import MPI 
from dolfinx.io    import gmshio
import dolfinx as dfx

# Load mesh
lf_list = np.asarray([1/(2**i) for i in range(0,7)]) #+[0.2,0.1,0.08,0.05]
dt_list = np.asarray([1/(2**i) for i in range(2,12)])
h_list  = []

t0      = 0      
deg_V   = 2
scheme  = sys.argv[1]
filename = f"Scheme = {scheme}"
answers = {}



for lf in lf_list:
    mesh_filename = "circle_in_Rectangle_LF" + str(lf) + ".msh"
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(r"meshes/"+mesh_filename, MPI.COMM_WORLD, rank = 0, gdim = 2)
    num_cells=mesh.topology.index_map(2).size_local
    h = dfx.cpp.mesh.h(mesh, 2, range(num_cells))
    h_min = h.min()
    h_list.append(h_min)
    Found_Working = False  
    for dt in dt_list:
        if Found_Working == True :
            answers[h_min,dt] = "blue"
            continue
        
        T       = np.max([dt*40,1]) # Ensure that enough dt is being done, even for coarse time discretizations

        solver = Solve_2D_cylinder_NS(mesh,
                                facet_tags,
                                identifier_string = filename,
                                scheme = scheme,
                                t  = t0,
                                T  = T,
                                dt = dt,
                                deg_V = deg_V)
        status = solver.Check_stability()
        if status == 'unstable':
            answers[h_min,dt] = "red"
        else :
            answers[h_min,dt] = "blue"
            Found_Working = True

plt.figure()
with open(f"{scheme}_stability.txt","w") as f:
    for i in range(len(lf_list)):
        for dt in dt_list:
            f.write(f"{dt};{lf_list[i]};{h_list[i]};{answers[h_list[i],dt]}\n")

for lf in lf_list:
    for dt in dt_list:
        if answers[h_min,dt] == "red":
            plt.plot([np.log2(lf)], [np.log2(dt)], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="red")
        else : 
            plt.plot([np.log2(lf)], [np.log2(dt)], marker="o", markersize=20, markeredgecolor="blue", markerfacecolor="blue")
plt.xlabel("h_min")
plt.ylabel("$\Delta t$")
plt.show()
plt.savefig(f'{scheme}_stability.png')

