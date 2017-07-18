import numpy as np
import operator as op
import functools as ft
import itertools as it
import more_itertools as mit

from .. import taskman as tm
from .. import diis
from . import closed as cls


# Public
def increment(n):
    return n + 1


def thresh_test(thresh):
    assert thresh > 0.

    def test(val):
        return abs(val) < thresh

    return test


def stop_if_all_true(*booleans):
    print(booleans)

    if all(booleans):
        raise StopIteration


def solve_rhf_mo_coefficients(s, h, g, n, d0=None, niter=100, e_thresh=1e-12,
                              r_thresh=1e-6, diis_start=2, diis_nvecs=1):
    fock = ft.partial(cls.fock, h, g)
    orbitals = ft.partial(cls.orbitals, s)
    density = ft.partial(cls.density, n)
    energy = ft.partial(cls.energy, h)
    orb_grad = ft.partial(cls.orb_grad, s)
    diis_step = ft.partial(diis.step, nmin=diis_start, nmax=diis_nvecs)

    f0 = fock(d0) if d0 is not None else h

    stream_start = {'f': f0, 'e0': 0., 'i': 0, 'diis_history': ()}
    step = tm.compose([
        tm.stream_modifier(f=increment, i='i', o='i'),
        tm.stream_modifier(f=orbitals, i='f', o='c'),
        tm.stream_modifier(f=density, i='c', o='d'),
        tm.stream_modifier(f=fock, i='d', o='f'),
        tm.stream_modifier(f=energy, i=('f', 'd'), o='e'),
        tm.stream_modifier(f=op.sub, i=('e', 'e0'), o='de'),
        tm.stream_modifier(i='e', o='e0'),
        tm.stream_modifier(f=orb_grad, i=('f', 'd'), o='w'),
        tm.stream_modifier(f=np.linalg.norm, i='w', o='r'),
        tm.stream_modifier(f=diis_step, i=('f', 'w', 'diis_history'),
                           o=('f', 'w', 'diis_history')),
        tm.stream_modifier(f=thresh_test(e_thresh), i='de', o='e_test'),
        tm.stream_modifier(f=thresh_test(r_thresh), i='r', o='r_test'),
        tm.stream_modifier(f=stop_if_all_true, i=('e_test', 'r_test'))
    ])
    stream_iterator = it.islice(mit.iterate(step, stream_start), niter + 1)

    stream_list = list(stream_iterator)
    print(stream_list[-1]['e_test'])
    stream_end = _last(stream_list)

    return stream_end


def solve_uhf_mo_coefficients():
    pass


def solve_rohf_mo_coefficients():
    pass


# Private
def _last(iterable):
    for value in iterable:
        pass
    return value
