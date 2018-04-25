import nuctools.mol as mol

from numpy.testing import assert_almost_equal

LABELS = ('O', 'H', 'H')
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))
LINEAR_COORDS = ((1.000000000000,  0.000000000000, 1.000000000000),
                 (1.000000000000,  1.638036840407, 1.000000000000),
                 (1.000000000000, -1.638036840407, 1.000000000000))
INERTIA = ((8.3401372297678691, 0., 0.),
           (0., 2.9318161492017873, 0.),
           (0., 0., 5.4083210805660809))
MOMENTS = (2.9318161492017873, 5.4083210805660809, 8.3401372297678691)
AXES = ((0., 0., 1.),
        (1., 0., 0.),
        (0., 1., 0.))


def test__masses():
    assert mol.masses(LABELS) == (15.99491461956, 1.00782503207, 1.00782503207)


def test__center_of_mass():
    cm = mol.center_of_mass(LABELS, COORDS)
    assert_almost_equal(cm, (0., 0., 0.), decimal=13)


def test__inertia_tensor():
    inertia = mol.inertia_tensor(LABELS, COORDS)
    assert_almost_equal(inertia, INERTIA, decimal=13)


def test__inertia_moments():
    moments = mol.inertia_moments(LABELS, COORDS)
    assert_almost_equal(moments, MOMENTS, decimal=13)


def test__inertia_axes():
    axes = mol.inertia_axes(LABELS, COORDS)
    assert_almost_equal(axes, AXES, decimal=13)


def test__is_linear():
    assert not mol.is_linear(COORDS)
    assert mol.is_linear(LINEAR_COORDS)


if __name__ == '__main__':
    test__is_linear()
