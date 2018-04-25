from ._isotopes import _ISOTOPIC_MASSES


def mass(label):
    """the mass of a nucleus

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :rtype: float
    """
    return _ISOTOPIC_MASSES[label.upper()]
