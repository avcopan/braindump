import more_itertools as mit


# Public
def ith(iterable, i, default=None):
    """ith item of an iterable, allowing negative indices

    :param iterable: an iterator
    :type iterable: typing.Iterator
    :param i: item index
    :type i: int
    :param default: return value when index is out of range
    :type default: object

    :rtype: object
    """
    if i >= 0:
        return mit.nth(iterable, n=i, default=default)
    else:
        return _neg_nth(iterable, n=i, default=default)


# Private
def _neg_nth(iterable, n, default=None):
    """nth item of an iterable, for n < 0
    """
    start = n
    stop = n - 1
    step = -1
    return next(mit.islice_extended(iterable, start, stop, step), default)
