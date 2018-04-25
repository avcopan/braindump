import numpy


def geometric_rank(coords, tol=None):
    """geometric rank of a system of nuclei

    0 = zero-dimensional (point), 1 = linear, 2 = planar, 3 = three-dimensional

    :param tol: threshold for identifying rank from SVD
    :type tol: float

    :rtype: int
    """
    x = numpy.subtract(coords, coords[0])
    return numpy.linalg.matrix_rank(x, tol=tol)
