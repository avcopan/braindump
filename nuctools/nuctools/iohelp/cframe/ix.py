from ._pairwise import iter_, iter_tril


def ix(n, w):
    """indices grouped by column frame

    :param n: number of rows/columns
    :type n: int
    :param w: column frame width
    :type w: int

    :rtype: tuple[tuple[int], tuple[int]]
    """
    return tuple(zip(*iter_(n, w)))


def ix_tril(n, w):
    """lower-triangle indices, grouped by column frame

    :param n: number of rows/columns
    :type n: int
    :param w: column frame width
    :type w: int

    :rtype: tuple[tuple[int], tuple[int]]
    """
    return tuple(zip(*iter_tril(n, w)))
