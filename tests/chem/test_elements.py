import pytest
from simplehf.chem import elements


def test__charge():
    assert elements.charge('uuo') == 118
    assert elements.charge('UUO') == 118
    assert elements.charge('UUO293') == 118
    assert elements.charge('UUO 293') == 118
    assert elements.charge(' \n UUO 293 \n ') == 118
    with pytest.raises(ValueError):
        elements.charge('293UUO')


def test__mass():
    assert elements.mass('uuo') == 293.21467
    assert elements.mass('UUO') == 293.21467
    assert elements.mass('UUO293') == 293.21467
    assert elements.mass('UUO 293') == 293.21467
    assert elements.mass(' \n UUO 293 \n ') == 293.21467
    with pytest.raises(ValueError):
        elements.mass('293UUO')


def test__atomic_symbol():
    assert elements.atomic_symbol('uuo') == 'UUO'
    assert elements.atomic_symbol('UUO') == 'UUO'
    assert elements.atomic_symbol('UUO293') == 'UUO'
    assert elements.atomic_symbol('UUO 293') == 'UUO'
    assert elements.atomic_symbol(' \n UUO 293 \n ') == 'UUO'
    with pytest.raises(ValueError):
        elements.atomic_symbol('293UUO')


def test__mass_number():
    assert elements.mass_number('uuo') is None
    assert elements.mass_number('UUO') is None
    assert elements.mass_number('UUO293') == 293
    assert elements.mass_number('UUO 293') == 293
    assert elements.mass_number(' \n UUO 293 \n ') == 293
    with pytest.raises(ValueError):
        elements.mass_number('293UUO')


def test__nuclear_label():
    assert elements.nuclear_label(symbol='uuo') == 'UUO'
    assert elements.nuclear_label(symbol='uuo', number=None) == 'UUO'
    assert elements.nuclear_label(symbol='uuo', number=293) == 'UUO293'
    assert elements.nuclear_label(symbol='uuo', number=293.) == 'UUO293'
    assert elements.nuclear_label(symbol='UUO', number=293.) == 'UUO293'
