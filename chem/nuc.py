import numpy
import itertools as it

from . import elements


# Public
def charges(labels: tuple) -> numpy.ndarray:
    """nuclear charges

    :param labels: nuclear labels
    :type labels: tuple

    :return: the charges
    :rtype: tuple
    """
    return tuple(map(elements.charge, labels))


def masses(labels: tuple) -> tuple:
    """nuclear masses

    :param labels: nuclear labels
    :type labels: tuple

    :return: the masses
    :rtype: tuple
    """
    return tuple(map(elements.mass, labels))


def energy(labels: tuple, coords: numpy.ndarray) -> float:
    """the coulomb energy of a system of nuclei

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray

    :return: the energy
    :rtype: float
    """
    return inverse_law(weights=charges(labels), coords=coords)


def moment(weights: tuple, coords: numpy.ndarray) -> tuple:
    """the weighted moment of a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray

    :return: the moment
    :rtype: tuple
    """
    return tuple(numpy.dot(weights, coords))


def center(weights: tuple, coords: numpy.ndarray) -> tuple:
    """the weighted center of a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray

    :return: the center
    :rtype: tuple
    """
    return tuple(numpy.divide(moment(weights, coords), sum(weights)))


def inverse_law(weights: tuple, coords: numpy.ndarray) -> float:
    """the weighted sum of inverse distances for a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: nuclear coordinates
    :type coords: numpy.ndarray

    :rtype: float
    """
    w_pairs = it.combinations(weights, r=2)
    r_pairs = it.combinations(coords, r=2)
    return sum(w1 * w2 / numpy.linalg.norm(numpy.subtract(r1, r2))
               for (w1, w2), (r1, r2) in zip(w_pairs, r_pairs))


# Testing
def _main():
    labels = ("O", "H", "H")
    coords = ((0.000000000000,  0.000000000000, -0.143225816552),
              (0.000000000000,  1.638036840407,  1.136548822547),
              (0.000000000000, -1.638036840407,  1.136548822547))
    print(energy(labels, coords))
    print(masses(labels))
    print(center(weights=charges(labels), coords=coords))


if __name__ == "__main__":
    _main()
