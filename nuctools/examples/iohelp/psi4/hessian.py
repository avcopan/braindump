import numpy
import nuctools

import os
from numpy.testing import assert_almost_equal


data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'files')
pattern = open(os.path.join(data_path, 'hessian.pattern')).read()
string = open(os.path.join(data_path, 'hessian.out')).read()
data = nuctools.iohelp.float_array(
        nuctools.iohelp.find(pattern, string))

ix = nuctools.iohelp.cframe.ix(n=9, w=5)
hessian = nuctools.math.array(shape=(9, 9), ix=ix, data=data)

hessian_ref = numpy.loadtxt(os.path.join(data_path, 'hessian.txt'))
assert_almost_equal(hessian, hessian_ref, decimal=10)


# PROJECTION
LABELS = ('o', 'h', 'h')
COORDS = ((0.0000000000,  0.0000000000, -0.1247219248),
          (0.0000000000, -1.4343021349,  0.9864370414),
          (0.0000000000,  1.4343021349,  0.9864370414))
coords = nuctools.mol.centered_coordinates(labels=LABELS, coords=COORDS)

axes = nuctools.mol.filtered_inertia_axes(labels=LABELS, coords=COORDS)


masses = nuctools.mol.masses(labels=LABELS)
e_t = nuctools.mol.translations(masses=masses, axes=axes)
e_r = nuctools.mol.rotations(masses=masses, coords=coords, axes=axes)
e_tr = numpy.hstack((e_t, e_r))

print(numpy.dot(e_tr.T, e_tr).round(10))

m = numpy.repeat(numpy.sqrt(masses), 3)
p = numpy.eye(*hessian.shape) - numpy.dot(e_tr, e_tr.T)
p = numpy.linalg.multi_dot([numpy.diag(m), p, numpy.diag(1./m)])

hess = numpy.linalg.multi_dot([p, hessian, p.T])

print(hess.round(3))
