
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
import ufl

from solutions import poiseuille
from elements import get_elements


def run_problem(N, V_element, Q_element):

    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # V_cg2 = ufl.VectorElement("CG", domain.ufl_cell(), 2)
    # Q_cg1 = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, V_element)
    Q = fem.FunctionSpace(domain, Q_element)

    mel = ufl.MixedElement([V_element, Q_element])
    W = fem.FunctionSpace(domain, mel)


    def u_boundary(x):
        return np.isclose(x[0], 0.0) | np.isclose(x[1], 1.0) | np.isclose(x[1], 0.0)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(poiseuille.u)
    u_facets = mesh.locate_entities_boundary(domain, fdim, u_boundary)
    u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, u_facets)
    bc0 = fem.dirichletbc(u_bc, u_dofs, W.sub(0))


    bcs = [bc0]
 
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(domain, ScalarType((0.0, 0.0)))

    a = fem.form( ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
                 + p * ufl.div(v) * ufl.dx + ufl.div(u) * q * ufl.dx )
    L = fem.form( ufl.inner(f, v) * ufl.dx )

    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    fem.petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")

    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
    pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)

    U = fem.Function(W)
    try:
        ksp.solve(b, U.vector)
    except PETSc.Error as e:
        if e.ierr == 92:
            print("The required PETSc solver/preconditioner is not available. Exiting.")
            print(e)
            exit(0)
        else:
            raise e

    uh, ph = U.sub(0).collapse(), U.sub(1).collapse()
    uh.name = "u_h"
    ph.name = "p_h"

    return uh, ph, domain

N = 60
element = "Mini"
V_el, Q_el = get_elements(element)

uh, ph, domain = run_problem(N, V_el, Q_el)



P1_vec_el = ufl.VectorElement("CG", domain.ufl_cell(), 1)
P1_vec = fem.FunctionSpace(domain, P1_vec_el)
P1_scal_el = ufl.FiniteElement("CG", domain.ufl_cell(), 1)
P1_scal = fem.FunctionSpace(domain, P1_scal_el)
uhp = fem.Function(P1_vec)
uhp.interpolate(uh)
uhp.name = "u_h"
php = fem.Function(P1_scal)
php.interpolate(ph)
# php = ph
php.vector[:] = php.vector[:]*-1 # Reset flipped sign of pressure
php.name = "p_h"

from dolfinx import io
with io.XDMFFile(domain.comm, "output/ex44_output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uhp)
    xdmf.write_function(php)


xx = domain.geometry.x
pp = php.vector.array
uh1 = uhp.vector.array[::2]


ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(xx[:,0], xx[:,1], pp, cmap=plt.cm.viridis)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("$p$")

ax = plt.figure().add_subplot(projection='3d')

ax.plot_trisurf(xx[:,0], xx[:,1], uh1, cmap=plt.cm.viridis)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("$u_1$")


plt.show()

