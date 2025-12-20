"""Atoms constants and utilities used across the package.

This module exposes a small set of public constants and a private helper
used throughout the package for identity mapping and small utilities.

Public API
- ``ATOMS_SYMBOLS_Z_TO_SYMBOL`` (Dict[int, str]): mapping atomic number -> element symbol.
- ``ATOMS_SYMBOLS_SYMBOL_TO_Z`` (Dict[str, int]): reverse mapping (symbol -> Z). Constructed from ``ATOMS_SYMBOLS_Z_TO_SYMBOL``.
- ``ANGULAR_MOMENTUM_MAP`` (Dict[str, int]): map angular momentum letter ("S","P",...) to numeric ``l``.
- ``EMPIRICAL_EXCEPTIONS`` (Dict[int, List[Dict[str,int]]]): experimentally verified ground-state electron configuration overrides for selected elements (atomic number -> list of subshell assignment dicts).

Private helpers
- ``_invert_dict``: internal utility to invert dictionaries while
  validating value uniqueness (raises ValueError on duplicates).
"""

from typing import Dict, List, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def _invert_dict(d: Dict[K, V]) -> Dict[V, K]:
    """Invert a dictionary mapping while validating value uniqueness.

    :param d: Dictionary to invert. All values must be unique.
    :type d: Dict[K, V]

    :returns: Inverted dictionary mapping original values -> original keys.
    :rtype: Dict[V, K]

    :raises ValueError: If the dictionary contains non-unique values and
        cannot be inverted.
    """
    if len(set(d.values())) != len(d):
        raise ValueError("Cannot invert dictionary with non-unique values")
    return {v: k for k, v in d.items()}


ATOMS_SYMBOLS_Z_TO_SYMBOL: Dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}

ATOMS_SYMBOLS_SYMBOL_TO_Z: Dict[str, int] = _invert_dict(
    d=ATOMS_SYMBOLS_Z_TO_SYMBOL
)

ANGULAR_MOMENTUM_MAP: Dict[str, int] = {
    "S": 0,
    "P": 1,
    "D": 2,
    "F": 3,
    "G": 4,
}

"""
Dictionary containing experimentally verified ground-state electron
configurations for elements whose true electron distributions deviate from
the Aufbau principle.

The data originates from the NIST Atomic Spectra Database (ASD), which
provides spectroscopically validated electron configurations for all
chemical elements. These empirical configurations incorporate real electron
correlation effects and subtle relativistic contributions that are not
captured by the theoretical (n + l) Aufbau ordering.

Each entry specifies a sequence of subshell-filling instructions expressed as
pure numeric quantum identifiers, without any string parsing:

    {
        "n": int,
        "l": int,
        "electron_count": int
    }

where:

    * n — principal quantum number
    * l — orbital angular momentum quantum number
    * electron_count — number of electrons occupying the (n,l) subshell

Reference
---------
NIST Atomic Spectra Database (ASD):
https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-8
https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A1%3A_Atomic_Electron_Configurations?utm_source=chatgpt.com
"""

EMPIRICAL_EXCEPTIONS: Dict[int, List[Dict[str, int]]] = {
    # Chromium (Cr), [Ar] 3d5 4s1
    24: [
        {"n": 3, "l": 2, "electron_count": 5},  # 3d5
        {"n": 4, "l": 0, "electron_count": 1},  # 4s1
    ],
    # Copper (Cu), [Ar] 3d10 4s1
    29: [
        {"n": 3, "l": 2, "electron_count": 10},  # 3d10
        {"n": 4, "l": 0, "electron_count": 1},  # 4s1
    ],
    # Niobium (Nb), [Kr] 4d4 5s1
    41: [
        {"n": 4, "l": 2, "electron_count": 4},  # 4d4
        {"n": 5, "l": 0, "electron_count": 1},  # 5s1
    ],
    # Molybdenum (Mo), [Kr] 4d5 5s1
    42: [
        {"n": 4, "l": 2, "electron_count": 5},  # 4d5
        {"n": 5, "l": 0, "electron_count": 1},  # 5s1
    ],
    # Ruthenium (Ru), [Kr] 4d7 5s1
    44: [
        {"n": 4, "l": 2, "electron_count": 7},  # 4d7
        {"n": 5, "l": 0, "electron_count": 1},  # 5s1
    ],
    # Rhodium (Rh), [Kr] 4d8 5s1
    45: [
        {"n": 4, "l": 2, "electron_count": 8},  # 4d8
        {"n": 5, "l": 0, "electron_count": 1},  # 5s1
    ],
    # Palladium (Pd), [Kr] 4d10
    46: [
        {"n": 4, "l": 2, "electron_count": 10},  # 4d10
    ],
    # Silver (Ag), [Kr] 4d10 5s1
    47: [
        {"n": 4, "l": 2, "electron_count": 10},  # 4d10
        {"n": 5, "l": 0, "electron_count": 1},  # 5s1
    ],
    # Platinum (Pt), [Xe] 4f14 5d9 6s1
    78: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f14
        {"n": 5, "l": 2, "electron_count": 9},  # 5d9
        {"n": 6, "l": 0, "electron_count": 1},  # 6s1
    ],
    # Gold (Au), [Xe] 4f14 5d10 6s1
    79: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f14
        {"n": 5, "l": 2, "electron_count": 10},  # 5d10
        {"n": 6, "l": 0, "electron_count": 1},  # 6s1
    ],
}
