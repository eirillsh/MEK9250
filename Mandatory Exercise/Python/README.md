## README ##
Author: Halvor Herlyng
Date  : March 13th, 2023

Running the python scripts requires a working installation of FEniCSx. A conda environment can be set up with the fenics-env.yml file within this directory, by running the following in a terminal:

conda env create -f fenicsx-env.yml


Activate the environment with:

conda activate fenicsx-env


Running the python scripts with a terminal where this environment is activated should work.



#####---------------------#####
#####---------------------#####


Exercise 1: 

The simulations can be run by running the python script 'convection_diffusion.py'. To choose between using standard Galerkin and SUPG, change the 'SUPG' variable at the top of the script:
False = standard Galerkin, True = SUPG


#####---------------------#####
#####---------------------#####


Exercise 2: 

Run simulations with the command 'python3 main.py int1 char int2',
where int1 has to be specified as an integer and char a character, and int2 is an integer. These control the Finite Element pairs of the discretization, the scheme used, and the degree of refinement of the mesh, respectively. Valid options:

int = 1 -> linear elements for both velocity and pressure space
int = 2 -> quadratic velocity elements, linear pressure elements (Taylor-Hood pair)

char = 'e' -> use explicit scheme with ICPS
char = 'i' -> use semi-implicit scheme

The coarseness of the mesh is chosen by specifying the length_factor variable in 'main.py'. A lower value means that the mesh is finer. The value chosen has to correspond to one of the values in the .msh files in the directory.

The initial time, final time and timestep size can be set in the 'main.py' file.


The file 'post_process.py' can be used to plot the pressure difference over the cylinder and the drag and lift coefficients as functions of time. Specify the case that should be plotted by setting the length_factor, timestep and scheme_type variables at the start of the script. The valid options for the length_factor variable are the numbers specified subsequent to "LF=" in the .msh filenames present in the directory. 

All the mesh files have been created with the 'create_mesh.py' file, where the mesh of the box with a cylindrical void can be created for a given "refinement factor" by adjusting the "length_factor" variable in the "Generate the mesh" section of the file. A lower length_factor yields a finer mesh.
