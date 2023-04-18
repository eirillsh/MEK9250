from dolfinx import fem, mesh

from mpi4py import MPI
from petsc4py import PETSc

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import ufl

def u_ex(x):
    sinx = ufl.sin(ufl.pi * x[0])
    siny = ufl.sin(ufl.pi * x[1])
    cosx = ufl.cos(ufl.pi * x[0])
    cosy = ufl.cos(ufl.pi * x[1])
    c_factor = 2 * ufl.pi * sinx * siny
    return c_factor * ufl.as_vector((cosy * sinx, - cosx * siny))

def p_ex(x):
    return ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

def source(x):
    u, p = u_ex(x), p_ex(x)
    return - ufl.div(ufl.grad(u)) + ufl.grad(p)


def create_bilinear_form(V, Q):
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    a_uu = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_up = ufl.inner(p, ufl.div(v)) * ufl.dx
    a_pu = ufl.inner(ufl.div(u), q) * ufl.dx
    return fem.form([[a_uu, a_up], [a_pu, None]])

def create_linear_form(V, Q):
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    domain = V.mesh
    x = ufl.SpatialCoordinate(domain)
    f = source(x)
    return fem.form([ufl.inner(f, v) * ufl.dx,
                     ufl.inner(fem.Constant(domain, 0.), q) * ufl.dx])


def create_velocity_bc(V):
    domain = V.mesh
    g = fem.Constant(domain, [0., 0.])
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, bdry_facets)
    return [fem.dirichletbc(g, dofs, V)]


def create_nullspace(rhs_form):
    null_vec = fem.petsc.create_vector_nest(rhs_form)
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0)
    null_vecs[1].set(1.0)
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    return nsp


def create_preconditioner(Q, a, bcs):
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    a_p11 = fem.form(ufl.inner(p, q) * ufl.dx)
    a_p = fem.form([[a[0][0], None],
                    [None, a_p11]])
    P = fem.petsc.assemble_matrix_nest(a_p, bcs)
    P.assemble()
    return P


def assemble_system(lhs_form, rhs_form, bcs):
    A = fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()

    b = fem.petsc.assemble_vector_nest(rhs_form)
    fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
    spaces = fem.extract_function_spaces(rhs_form)
    bcs0 = fem.bcs_by_block(spaces, bcs)
    fem.petsc.set_bc_nest(b, bcs0)
    return A, b


def create_block_solver(A, b, P, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]),
                                ("p", nested_IS[0][1]))

    # Set the preconditioners for each block
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    return ksp


def assemble_scalar(J, comm: MPI.Comm):
    scalar_form = fem.form(J)
    local_J = fem.assemble_scalar(scalar_form)
    return comm.allreduce(local_J, op=MPI.SUM)

def compute_errors(u, p):
    domain = u.function_space.mesh
    x = ufl.SpatialCoordinate(domain)
    error_u = u - u_ex(x)
    H1_u = ufl.inner(error_u, error_u) * ufl.dx
    H1_u += ufl.inner(ufl.grad(error_u), ufl.grad(error_u)) * ufl.dx
    velocity_error = np.sqrt(assemble_scalar(H1_u, domain.comm))

    error_p = -p - p_ex(x)
    L2_p = fem.form(error_p * error_p * ufl.dx)
    pressure_error = np.sqrt(assemble_scalar(L2_p, domain.comm))
    return velocity_error, pressure_error


def solve_stokes(u_element, p_element, domain):
    V = fem.FunctionSpace(domain, u_element)
    Q = fem.FunctionSpace(domain, p_element)

    lhs_form = create_bilinear_form(V, Q)
    rhs_form = create_linear_form(V, Q)

    bcs = create_velocity_bc(V)
    nsp = create_nullspace(rhs_form)
    A, b = assemble_system(lhs_form, rhs_form, bcs)
    assert nsp.test(A)
    A.setNullSpace(nsp)

    P = create_preconditioner(Q, lhs_form, bcs)
    ksp = create_block_solver(A, b, P, domain.comm)

    u, p = fem.Function(V), fem.Function(Q)
    w = PETSc.Vec().createNest([u.vector, p.vector])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()
    # return compute_errors(u, p)
    return u, p, V, Q


def main():

    from timeit import default_timer as timer
    from dolfinx import log

    N = 128
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    u_element = ufl.VectorElement("Lagrange", domain.ufl_cell(), 3)
    p_element = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 2)

    start = timer()
    uh, ph, V, Q = solve_stokes(u_element, p_element, domain)
    end = timer()

    print(f"{N=}, {(end-start)*1e3:.3f} ms")

    P1_vec_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    P1_vec = fem.FunctionSpace(domain, P1_vec_el)
    P1_scal_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    P1_scal = fem.FunctionSpace(domain, P1_scal_el)
    uhp = fem.Function(P1_vec)
    uhp.interpolate(uh)
    uhp.name = "u_h"
    php = fem.Function(P1_scal)
    php.interpolate(ph)
    php.vector[:] = php.vector[:]*-1 # Reset flipped sign of pressure
    php.name = "p_h"

    from dolfinx import io
    with io.XDMFFile(domain.comm, "output/bp-stokes_output.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uhp)
        xdmf.write_function(php)


    return


if __name__ == '__main__':
    main()
