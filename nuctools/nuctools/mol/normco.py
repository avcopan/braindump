import numpy
import functools as ft

from ..math import normalize_columns
from ..math import column_vector


# Public
def translations(masses, axes):
    """unit vectors for translation

    :param masses: nuclear masses by atom
    :param axes: a 3x3 matrix of Cartesian axes (probably [[1, 0, 0], ...])

    :return: translation vectors
    :rtype: numpy.ndarray
    """
    w = column_vector(numpy.sqrt(masses))
    trans_vecs = numpy.kron(w, axes)
    return normalize_columns(trans_vecs)


def rotations(masses, coords, axes):
    """unit vectors for rotation

    :param masses: nuclear masses by atom
    :param axes: a 3x3 matrix of Cartesian axes (probably [[1, 0, 0], ...])

    :return: translation vectors
    :rtype: numpy.ndarray
    """
    w = column_vector(numpy.sqrt(masses))
    x = coords  # numpy.dot(coords, axes)
    xw = numpy.multiply(x, w)
    cross_xw = ft.partial(numpy.cross, xw)
    rot_vecs = numpy.hstack(map(column_vector, map(cross_xw, axes.T)))
    return normalize_columns(rot_vecs)
