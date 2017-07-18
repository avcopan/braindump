import numpy as np
from . import linalg


# Public
def expectation_value(o, ad, axes=(0, 1)):
    return linalg.matmul_trace(o, ad + ad, axes=axes)


def energy(h, af, ad):
    return linalg.matmul_trace(h + af, ad + ad) / 2.


def mean_field(g, ad):
    j = linalg.matmul_trace(g, ad + ad, axes=(1, 3))
    ak = linalg.matmul_trace(g, ad, axes=(1, 2))
    return j - ak


def fock(h, g, ad):
    return h + mean_field(g, ad)


def orbitals(s, af):
    return linalg.eigh_vectors(s, af)


def density(na, ac):
    ac_o = ac[:, :na]
    return linalg.outer_square(ac_o)


def orb_grad(s, af, ad):
    return np.linalg.multi_dot([af, ad, s]) - np.linalg.multi_dot([s, ad, af])
