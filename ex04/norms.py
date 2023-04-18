import numpy as np

from mpi4py import MPI
from dolfinx import fem
import ufl

def compute_H10_norm(uh):

    H1_sqr_form = fem.form( ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H10_error(uh, uex):
    
    H1_sqr_form = fem.form( ufl.dot(ufl.grad(uh-uex), ufl.grad(uh-uex)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H1_norm(uh):

    H1_sqr_form = fem.form( ufl.dot(uh, uh) * ufl.dx + \
                            ufl.dot(ufl.grad(uh), ufl.grad(uh)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_H1_error(uh, uex):
    
    H1_sqr_form = fem.form( ufl.dot(uh-uex, uh-uex) * ufl.dx + \
                            ufl.dot(ufl.grad(uh-uex), ufl.grad(uh-uex)) * ufl.dx )
    H1_local = fem.assemble_scalar(H1_sqr_form)
    H1 = np.sqrt(uh.function_space.mesh.comm.allreduce(H1_local, op=MPI.SUM))

    return H1

def compute_L2_norm(uh):

    L2_sqr_form = fem.form( ufl.inner(uh, uh) * ufl.dx )
    L2_local = fem.assemble_scalar(L2_sqr_form)
    L2 = np.sqrt(uh.function_space.mesh.comm.allreduce(L2_local, op=MPI.SUM))

    return L2

def compute_L2_error(uh, uex):

    L2_sqr_form = fem.form( ufl.dot(uh-uex, uh-uex) * ufl.dx )
    L2_local = fem.assemble_scalar(L2_sqr_form)
    L2 = np.sqrt(uh.function_space.mesh.comm.allreduce(L2_local, op=MPI.SUM))

    return L2

