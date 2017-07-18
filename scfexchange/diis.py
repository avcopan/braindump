import warnings

import numpy as np
import functools as ft
import itertools as it
import scipy.linalg as spla


# Public
def update(entry, history, nmax):
    return tuple(history)[-(nmax - 1):] + (entry,)


def step(p, r, history, nmin=2, nmax=6):
    assert all(len(pair) is 2 for pair in history)

    series = update((p, r), history, nmax)
    p_series, r_series = zip(*series)

    if len(r_series) >= nmin:
        c = coefficients(r_series)
    else:
        c = _standard_unit_vector(len(r_series), -1)

    p_extrap = _linear_combination(p_series, c)
    r_extrap = _linear_combination(r_series, c)

    return p_extrap, r_extrap, series


def coefficients(r_series):
    assert len(r_series) >= 2
    a = a_matrix(r_series)
    b = b_vector(len(r_series))
    with warnings.catch_warnings(record=True):
        x = spla.solve(a, b)
    return x[:-1]


def a_matrix(r_series):
    n = len(r_series)
    a = np.zeros((n + 1, n + 1))
    r_pairs = it.combinations_with_replacement(r_series, r=2)
    a[np.triu_indices(n)] = tuple(it.starmap(_vdot, r_pairs))
    a[np.tril_indices(n, k=-1)] = a.T[np.tril_indices(n, k=-1)]
    a[n, range(n)] = a[range(n), n] = -1
    return a


def b_vector(n):
    b = np.zeros((n + 1,))
    b[n] = -1
    return b


# Private
def _vdot(p1, p2):
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return np.vdot(p1, p2)
    try:
        return sum(it.starmap(np.vdot, zip(p1, p2)))
    except:
        raise Exception("Could not align array collections for vdot.")


def _standard_unit_vector(length, axis):
    e = np.zeros((length,))
    e[axis] = 1.
    return e


def _linear_combination(series, weights):
    assert len(series) is len(weights)
    weight = ft.partial(np.tensordot, weights, axes=(0, 0))
    return tuple(map(weight, zip(*series)))
