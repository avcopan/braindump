from numpy.testing import assert_equal, assert_almost_equal
from simplehf.chem import nuc

LABELS = ("O", "H", "H")
COORDS = ((0.000000000000,  0.000000000000, -0.143225816552),
          (0.000000000000,  1.638036840407,  1.136548822547),
          (0.000000000000, -1.638036840407,  1.136548822547))

ENERGY = 8.00236706181
CHARGES = (8, 1, 1)
MASSES = (15.99491461956, 1.00782503207, 1.00782503207)
CHARGE_MOMENT = (0.0, 0.0, 1.1272911126779999)
CHARGE_CENTER = (0.0, 0.0, 0.11272911126779998)


def test__energy():
    energy = nuc.energy(LABELS, COORDS)
    assert_almost_equal(energy, ENERGY, decimal=10)


def test__charges():
    charges = nuc.charges(LABELS)
    assert_equal(charges, CHARGES)


def test__masses():
    masses = nuc.masses(LABELS)
    assert_almost_equal(masses, MASSES, decimal=10)


def test__moment():
    charge_moment = nuc.moment(weights=CHARGES, coords=COORDS)
    assert_almost_equal(charge_moment, CHARGE_MOMENT, decimal=10)


def test__center():
    charge_center = nuc.center(weights=CHARGES, coords=COORDS)
    assert_almost_equal(charge_center, CHARGE_CENTER, decimal=10)


def test__inverse_law():
    energy = nuc.inverse_law(weights=CHARGES, coords=COORDS)
    assert_almost_equal(energy, ENERGY, decimal=10)
