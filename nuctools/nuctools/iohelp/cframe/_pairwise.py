import math
import itertools as it


def iter_tril(n, w):
    """iterate lower-triangle index pairs by column frame

    :param n: number of rows/columns
    :type n: int
    :param w: column frame width
    :type w: int

    :rtype: typing.Iterator[tuple[int, int]]
    """
    return tril_filter(iter_(n, w))


def iter_(n, w):
    """iterate index pairs by column frame

    :param n: number of rows/columns
    :type n: int
    :param w: column frame width
    :type w: int

    :rtype: typing.Iterator[tuple[int, int]]
    """
    nframe = math.ceil(n / w)
    return it.chain(*(iter_ith_cframe(i, n, w) for i in range(nframe)))


def iter_ith_cframe(i, n, w):
    """iterate index pairs for the ith column frame

    :param i: column frame index
    :type i: int
    :param n: number of rows/columns
    :type n: int
    :param w: column frame width
    :type w: int

    :rtype: typing.Iterator[tuple[int, int]]
    """
    minrow = 0
    maxrow = n
    mincol = i * w
    maxcol = min(n, (i + 1) * w)
    return it.product(range(minrow, maxrow), range(mincol, maxcol))


def tril_filter(pairs):
    """iterate over (i, j) pairs satisfying i >= j
    """
    return filter(lambda pair: pair[0] >= pair[1], pairs)
