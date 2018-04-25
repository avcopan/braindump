import numpy
import scipy.linalg as spla

from . import nuc
from .geom import geometric_rank
from ..math import column_vector


def masses(labels):
    """nuclear masses

    :param labels: nuclear labels
    :type labels: tuple

    :return: the masses
    :rtype: tuple
    """
    return tuple(map(nuc.mass, labels))


def center_of_mass(labels, coords):
    """center of mass

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    m = masses(labels)
    return _center(weights=m, coords=coords)


def centered_coordinates(labels, coords):
    """coordinates relative to the center of mass

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    cm = center_of_mass(labels=labels, coords=coords)
    return numpy.subtract(coords, cm)


def inertia_tensor(labels, coords):
    """inertia tensor about the center of mass

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    x = centered_coordinates(labels=labels, coords=coords)
    return _inertia_tensor(labels=labels, coords=x)


def inertia_axes(labels, coords):
    """inertial axes about the center of mass

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    inertia = inertia_tensor(labels=labels, coords=coords)
    _, axes = spla.eigh(inertia)
    return axes


def inertia_moments(labels, coords):
    """moments of inertia about the center of mass

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    inertia = inertia_tensor(labels=labels, coords=coords)
    moments, _ = spla.eigh(inertia)
    return moments


def filtered_inertia_axes(labels, coords, tol=None):
    """inertial axes about the center of mass, filtered by geometric rank

    Returns only the non-zero axes for linear molecules.  For points (single
    atoms) this returns an empty array.

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray
    :param tol: threshold for identifying linearity from singular values
    :type tol: float

    :rtype: numpy.ndarray
    """
    rank = geometric_rank(coords=coords, tol=tol)
    slc = slice(0) if rank is 0 else slice(1, 3) if rank is 1 else slice(None)
    axes = inertia_axes(labels=labels, coords=coords)
    return axes[:, slc]


def filtered_inertia_moments(labels, coords, tol=None):
    """inertial axes about the center of mass, filtered by geometric rank

    Returns only the non-zero moments for linear molecules.  For points
    (single atoms) this returns an empty array.

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray
    :param tol: threshold for identifying linearity from singular values
    :type tol: float

    :rtype: numpy.ndarray
    """
    rank = geometric_rank(coords=coords, tol=tol)
    slc = slice(0) if rank is 0 else slice(1, 3) if rank is 1 else slice(None)
    moments = inertia_moments(labels=labels, coords=coords)
    return moments[slc]


# Private
def _inertia_tensor(labels, coords):
    """inertia tensor about the origin

    :param labels: nuclear labels
    :type labels: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    m = column_vector(masses(labels))
    mx = numpy.multiply(coords, m)
    mxx = numpy.dot(numpy.transpose(mx), coords)
    return numpy.trace(mxx) * numpy.eye(3) - mxx


def _center(weights, coords):
    """the weighted center of a system of nuclei

    :param weights: weights
    :type weights: tuple
    :param coords: coordinates
    :type coords: numpy.ndarray

    :rtype: numpy.ndarray
    """
    mu = numpy.array(numpy.dot(weights, coords))
    return tuple(numpy.divide(mu, sum(weights)))
