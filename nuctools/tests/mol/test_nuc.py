import nuctools.mol.nuc as nuc


def test__mass():
    assert nuc.mass('uuo') == 293.21467
    assert nuc.mass('uuo293') == 293.21467
