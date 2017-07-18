from . import linalg


def expectation_value(o, ad, bd, axes=(0, 1)):
    return linalg.matmul_trace(o, ad + bd, axes=axes)


def energy(h, af, bf, ad, bd):
    ae = linalg.matmul_trace(h + af, ad)
    be = linalg.matmul_trace(h + bf, bd)
    return (ae + be) / 2.


def mean_field(g, ad, bd):
    j = linalg.matmul_trace(g, ad + bd, axes=(1, 3))
    ak = linalg.matmul_trace(g, ad, axes=(1, 2))
    bk = linalg.matmul_trace(g, bd, axes=(1, 2))
    return j - ak, j - bk


def fock(h, g, ad, bd):
    aw, bw = mean_field(g, ad, bd)
    return h + aw, h + bw


def mo_coefficients(s, af, bf):
    return linalg.eigh_vectors(s, af), linalg.eigh_vectors(s, bf)


def density(na, nb, ac, bc):
    ad = linalg.outer_square(ac[:, :na])
    bd = linalg.outer_square(bc[:, :nb])
    return ad, bd
