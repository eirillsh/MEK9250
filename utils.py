import dolfin as df
import sympy as sp
import numpy as np


x, y, z = sp.symbols("x[0] x[1] x[2]")


def to_expression(f, degree=5):
    if isinstance(f, sp.Expr): f = f.simplify()
    f_c = sp.printing.ccode(f).replace("M_PI", "pi").replace("log", "std::log")
    return df.Expression(f_c, degree=degree)


def L2_norm(u):
    f = u*u
    I = df.assemble(f*df.dx)
    return df.sqrt(I)


def H1_norm(u):
    f = u*u + df.inner(df.grad(u), df.grad(u))
    I = df.assemble(f*df.dx)
    return df.sqrt(I)
