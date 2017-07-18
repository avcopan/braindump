import numpy as np
import scipy.linalg as spla


def eigh_vectors(s, f):
    _, c = spla.eigh(f, b=s)
    return c


def outer_square(c):
    return np.dot(c, c.T)


def matmul_trace(o, d, axes=(0, 1)):
    return np.tensordot(o, d, axes=[axes, (1, 0)])
