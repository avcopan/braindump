import more_itertools as mit


def nth(iterable, n, default=None):
    """nth item of an iterable, allowing negative indices

    :param iterable: an iterator
    :type iterable: typing.Iterator
    :param n: item index
    :type n: int
    :param default: return value when index is out of range
    :type default: object

    :rtype: object
    """
    if n >= 0:
        return mit.nth(iterable, n=n, default=default)
    else:
        return _minus_nth(iterable, n=n, default=default)


def _minus_nth(iterable, n, default=None):
    """nth item of an iterable, for negative indices only
    """
    start = n
    stop = n - 1
    step = -1
    return next(mit.islice_extended(iterable, start, stop, step), default)


if __name__ == '__main__':
    print(nth(range(10), n=-2))
    print(nth(range(10), n=-12))
