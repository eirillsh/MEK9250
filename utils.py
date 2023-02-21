import dolfin as df
import sympy as sp
import numpy as np


x, y, z = sp.symbols("x[0] x[1] x[2]")


def to_expression(f, degree=5):
    if isinstance(f, (int, float)): 
        return df.Constant(f)
    elif isinstance(f, str):
        return df.Expression(f, degree=degree)
    elif isinstance(f, sp.Expr):
        f_c = sp.printing.ccode(f.simplify())
        f_c = f_c.replace("M_PI", "pi").replace("log", "std::log")
        return df.Expression(f_c, degree=degree)
    else:
        raise TypeError("Invalid type for expression")


def solve(mesh, degree, a, L, bc):
    '''Solve using Lagrange polynomials.'''
    V = df.FunctionSpace(mesh, "CG", degree)
    psi = df.TrialFunction(V)
    phi = df.TestFunction(V)
    u = df.Function(V)
    df.solve(a(psi, phi) == L(phi), u, bc(V))
    return u


def boundary(x, on_boundary):
    return on_boundary


def L2_norm(u):
    f = u*u
    I = df.assemble(f*df.dx)
    return df.sqrt(I)


def H1_norm(u):
    f = u*u + df.inner(df.grad(u), df.grad(u))
    I = df.assemble(f*df.dx)
    return df.sqrt(I)

def H1_seminorm(u):
    f = df.inner(df.grad(u), df.grad(u))
    I = df.assemble(f*df.dx)
    return df.sqrt(I)

