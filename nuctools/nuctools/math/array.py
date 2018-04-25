import numpy


def array(shape, ix, data):
    """initialize an array and fill it with data

    ```
    >>> a = numpy.zeros(shape)
    >>> a[ix] = data

    ```

    :param shape: array shape
    :type shape: tuple
    :param ix: indices for data
    :type ix: tuple[numpy.ndarray, ...]
    :param data: data
    :type data: Iterable

    :rtype: numpy.ndarray
    """
    a = numpy.zeros(shape)
    a[ix] = data
    return a
