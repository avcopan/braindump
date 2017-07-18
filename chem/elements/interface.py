"""Provides an interface for element and isotope data.
"""
import re
from ._elements import _ATOMIC_CHARGES
from ._isotopes import _ISOTOPIC_MASSES


# Public
def charge(label: str) -> int:
    """the charge of a nucleus

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :return: the charge
    :rtype: int
    """
    symbol = atomic_symbol(label)
    return _ATOMIC_CHARGES[symbol]


def mass(label: str) -> float:
    """the mass of a nucleus

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :return: the mass
    :rtype: float
    """
    symbol = atomic_symbol(label)
    number = mass_number(label)
    nuc_label = nuclear_label(symbol, number)
    return _ISOTOPIC_MASSES[nuc_label]


def atomic_symbol(label: str) -> str:
    """the atomic symbol of a nucleus

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :return: the atomic symbol
    :rtype: str
    """
    symbol = _nuclear_label_dict(label)['symbol']
    return symbol.upper()


def mass_number(label: str) -> int:
    """the mass number of a nucleus

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :return: the mass number
    :rtype: int
    """
    number_str = _nuclear_label_dict(label)['number']
    number = int(number_str) if number_str is not '' else None
    return number


def nuclear_label(symbol: str, number: int=None) -> str:
    """the label of an element or isotope

    :param symbol: an atomic symbol
    :type symbol: str
    :param number: a mass number, or :obj:`None`
    :type number: int

    :return: the label
    :rtype: str
    """
    cap_symbol = symbol.upper()
    return cap_symbol if number is None else cap_symbol + str(int(number))


# Private
_NUCLEAR_LABEL_PATTERN = r'(?P<symbol>[A-Za-z]{1,3})\s*(?P<number>[0-9]{0,3})'


def _nuclear_label_dict(label: str) -> dict:
    """dictionary containing the symbol and number in a label

    :param label: an atomic symbol, possibly followed by a mass number
    :type label: str

    :return: a dictionary of strings, with keys :obj:`symbol` and :obj:`number`
    :rtype: dict
    """
    match = re.fullmatch(_NUCLEAR_LABEL_PATTERN, label.strip())
    if not match:
        raise ValueError("invalid `label` argument")
    return match.groupdict()


# Testing
def _main():
    print(charge("UUO"))
    print(charge("UUO293"))
    print(mass("UUO"))
    print(mass("UUO293"))


if __name__ == "__main__":
    _main()
