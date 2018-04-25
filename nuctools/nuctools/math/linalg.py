import numpy


def normalize_columns(m):
    """normalize the columns of a matrix

    :param m: matrix
    :type m: numpy.ndarray

    :rtype: numpy.ndarray
    """
    norms = list(map(numpy.linalg.norm, numpy.transpose(m)))
    return numpy.divide(m, norms)


def column_vector(v):
    """form an n x 1 column vector, flattening if necessary

    :param v: vector or array
    :type v: numpy.ndarray

    :rtype: numpy.ndarray
    """
    return numpy.reshape(numpy.ravel(v), (-1, 1))
