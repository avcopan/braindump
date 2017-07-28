import more_itertools as mit
from sympy.combinatorics.permutations import _af_parity as parity


# Public
def riffle_shuffles(iterable, comp):
    cshuffle = card_shuffler(deck_sizes=comp)
    rshuffle = permuter(iterable)

    dshuffles = deck_shuffles(deck_sizes=comp)
    cshuffles = map(cshuffle, dshuffles)

    return map(lambda pi: (_sgn(pi), rshuffle(pi)), cshuffles)


# Private
def _sgn(pi):
    return (-1) ** parity(pi)


def deck_shuffles(deck_sizes):
    decks = sum((size * (deck,) for deck, size in enumerate(deck_sizes)), ())
    return mit.distinct_permutations(decks)


def card_shuffler(deck_sizes):

    def card_shuffle(deck_shuffle):
        return tuple((deck_shuffle[:p].count(d) + sum(deck_sizes[:d])
                      for p, d in enumerate(deck_shuffle)))

    return card_shuffle


def permuter(iterable):

    pool = tuple(iterable)

    def permutation(pi):
        return tuple(pool[i] for i in pi)

    return permutation


if __name__ == '__main__':
    print(tuple(riffle_shuffles('abcd', (2, 2))))
