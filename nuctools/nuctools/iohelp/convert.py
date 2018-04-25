import numpy


def float_array(iterable):
    """convert an iterable of objects to an array of floats

    :param iterable: an iterable
    :type iterable: typing.Iterable[object, ...]

    :rtype: numpy.ndarray
    """
    return numpy.array(list(map(float, iterable)))
