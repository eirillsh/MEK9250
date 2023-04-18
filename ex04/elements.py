import ufl
from dolfinx import mesh

def P_el_builder(n_V, n_Q):
    V_type = "CG" if n_V > 0 else "DG"
    Q_type = "CG" if n_Q > 0 else "DG"
    V_el = ufl.VectorElement(V_type, "triangle", n_V)
    Q_el = ufl.FiniteElement(Q_type, "triangle", n_Q)

    return V_el, Q_el

def CR_builder():
    V_type = "Crouzeix-Raviart"
    Q_type = "DG"
    V_el = ufl.VectorElement(V_type, "triangle", 1)
    Q_el = ufl.FiniteElement(Q_type, "triangle", 0)

    return V_el, Q_el

def mini_builder():

    V_el = ufl.VectorElement(
        ufl.EnrichedElement(
            ufl.FiniteElement("CG", "triangle", 1),
            ufl.FiniteElement("Bubble", "triangle", 3)
        )
    )
    Q_el = ufl.FiniteElement("CG", "triangle", 1)

    return V_el, Q_el

def get_elements(element:str):

    if element == "P2-P1":
        return P_el_builder(2, 1)
    elif element == "P2-P0":
        return P_el_builder(2, 0)
    elif element == "P1-P0":
        return P_el_builder(2, 0)
    elif element == "P2-P2":
        return P_el_builder(2, 2)
    elif element == "P1-P1":
        return P_el_builder(1, 1)
    elif element == "CR":
        return CR_builder()
    elif element == "Mini":
        return mini_builder()
    else:
        raise ValueError("Element not recognized.")


