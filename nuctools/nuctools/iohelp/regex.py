import re
import functools as ft
from ._ith import ith


def find(pattern, string, i=0, flags=re.MULTILINE):
    """find the ith non-overlapping match of pattern in string

    equivalent to re.findall(pattern, string, flags=flags)[i], but doesn't
    need to cache captured instances and defaults to `None` when nothing is
    found

    :param pattern: re pattern
    :type pattern: str
    :param string: string to search
    :type string: str
    :param i: positive or negative index (-1 finds the last match)
    :type i: int
    :param flags: re flags
    :type flags: int
    :param default: default value, when index is out of range
    :type default: object

    :rtype: object
    """
    return capture(ith(re.finditer(pattern, string, flags=flags), i=i))


@ft.singledispatch
def capture(match):
    """imitates the behavior of re.findall for matches
    """
    groups = match.groups()
    return (match.group(0) if len(groups) == 0 else
            match.group(1) if len(groups) == 1 else groups)


@capture.register(type(None))
def _(match):
    return None
