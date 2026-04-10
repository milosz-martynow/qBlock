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

"""Dictionary containing experimentally verified ground-state electron
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

    * n  -- principal quantum number
    * l  -- orbital angular momentum quantum number encoded as an integer
    * electron_count -- number of electrons occupying the subshell
"""

EMPIRICAL_EXCEPTIONS: Dict[int, List[Dict[str, int]]] = {
    # Chromium: [Ar] 3d^5 4s^1 instead of [Ar] 3d^4 4s^2
    24: [
        {"n": 3, "l": 2, "electron_count": 5},  # 3d^5
        {"n": 4, "l": 0, "electron_count": 1},  # 4s^1
    ],
    # Copper: [Ar] 3d^10 4s^1 instead of [Ar] 3d^9 4s^2
    29: [
        {"n": 3, "l": 2, "electron_count": 10},  # 3d^10
        {"n": 4, "l": 0, "electron_count": 1},  # 4s^1
    ],
    # Niobium: [Kr] 4d^4 5s^1 instead of [Kr] 4d^3 5s^2
    41: [
        {"n": 4, "l": 2, "electron_count": 4},  # 4d^4
        {"n": 5, "l": 0, "electron_count": 1},  # 5s^1
    ],
    # Molybdenum: [Kr] 4d^5 5s^1 instead of [Kr] 4d^4 5s^2
    42: [
        {"n": 4, "l": 2, "electron_count": 5},  # 4d^5
        {"n": 5, "l": 0, "electron_count": 1},  # 5s^1
    ],
    # Ruthenium: [Kr] 4d^7 5s^1 instead of [Kr] 4d^6 5s^2
    44: [
        {"n": 4, "l": 2, "electron_count": 7},  # 4d^7
        {"n": 5, "l": 0, "electron_count": 1},  # 5s^1
    ],
    # Rhodium: [Kr] 4d^8 5s^1 instead of [Kr] 4d^7 5s^2
    45: [
        {"n": 4, "l": 2, "electron_count": 8},  # 4d^8
        {"n": 5, "l": 0, "electron_count": 1},  # 5s^1
    ],
    # Palladium: [Kr] 4d^10 instead of [Kr] 4d^8 5s^2
    46: [
        {"n": 4, "l": 2, "electron_count": 10},  # 4d^10
    ],
    # Silver: [Kr] 4d^10 5s^1 instead of [Kr] 4d^9 5s^2
    47: [
        {"n": 4, "l": 2, "electron_count": 10},  # 4d^10
        {"n": 5, "l": 0, "electron_count": 1},  # 5s^1
    ],
    # Platinum: [Xe] 4f^14 5d^9 6s^1 instead of [Xe] 4f^14 5d^8 6s^2
    78: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f^14
        {"n": 5, "l": 2, "electron_count": 9},  # 5d^9
        {"n": 6, "l": 0, "electron_count": 1},  # 6s^1
    ],
    # Gold: [Xe] 4f^14 5d^10 6s^1 instead of [Xe] 4f^14 5d^9 6s^2
    79: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f^14
        {"n": 5, "l": 2, "electron_count": 10},  # 5d^10
        {"n": 6, "l": 0, "electron_count": 1},  # 6s^1
    ],
}


# ==============================================================================
# Closed-shell and open-shell atom classification
# ==============================================================================
#
# These lists classify all atoms (Z=1 to Z=118) based on their ground-state
# electron configuration. An atom is:
#   - Closed-shell: All electrons are paired (no unpaired electrons)
#   - Open-shell: Has at least one unpaired electron
#
# The classification accounts for:
#   - Aufbau principle filling
#   - Hund's rule (parallel spins in degenerate orbitals)
#   - Empirical exceptions from NIST data
#
# Note: For atoms with empirical exceptions, the classification is based on
# the actual (empirical) configuration, not the theoretical Aufbau prediction.
# ==============================================================================

# Closed-shell atoms: all electrons are paired
# Includes noble gases and atoms with completely filled subshells
CLOSED_SHELL_ATOMS: List[int] = [
    # Noble gases (full outer shells)
    2,  # He: 1s²
    10,  # Ne: [He] 2s² 2p⁶
    18,  # Ar: [Ne] 3s² 3p⁶
    36,  # Kr: [Ar] 3d¹⁰ 4s² 4p⁶
    54,  # Xe: [Kr] 4d¹⁰ 5s² 5p⁶
    86,  # Rn: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁶
    118,  # Og: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁶
    # Alkaline earth metals (ns²)
    4,  # Be: [He] 2s²
    12,  # Mg: [Ne] 3s²
    20,  # Ca: [Ar] 4s²
    38,  # Sr: [Kr] 5s²
    56,  # Ba: [Xe] 6s²
    88,  # Ra: [Rn] 7s²
    # Group 12 (d¹⁰ s²)
    30,  # Zn: [Ar] 3d¹⁰ 4s²
    48,  # Cd: [Kr] 4d¹⁰ 5s²
    80,  # Hg: [Xe] 4f¹⁴ 5d¹⁰ 6s²
    112,  # Cn: [Rn] 5f¹⁴ 6d¹⁰ 7s²
    # Palladium (empirical exception: 4d¹⁰, no 5s electrons)
    46,  # Pd: [Kr] 4d¹⁰
    # Ytterbium (f¹⁴ s²)
    70,  # Yb: [Xe] 4f¹⁴ 6s²
    # Nobelium (f¹⁴ s²)
    102,  # No: [Rn] 5f¹⁴ 7s²
]

# Open-shell atoms: have at least one unpaired electron
# Includes all atoms not in CLOSED_SHELL_ATOMS
OPEN_SHELL_ATOMS: List[int] = [
    # Hydrogen
    1,  # H: 1s¹ (1 unpaired)
    # Alkali metals (ns¹)
    3,  # Li: [He] 2s¹
    11,  # Na: [Ne] 3s¹
    19,  # K: [Ar] 4s¹
    37,  # Rb: [Kr] 5s¹
    55,  # Cs: [Xe] 6s¹
    87,  # Fr: [Rn] 7s¹
    # Group 13 (p¹)
    5,  # B: [He] 2s² 2p¹
    13,  # Al: [Ne] 3s² 3p¹
    31,  # Ga: [Ar] 3d¹⁰ 4s² 4p¹
    49,  # In: [Kr] 4d¹⁰ 5s² 5p¹
    81,  # Tl: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p¹
    113,  # Nh: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p¹
    # Group 14 (p²)
    6,  # C: [He] 2s² 2p² (2 unpaired)
    14,  # Si: [Ne] 3s² 3p²
    32,  # Ge: [Ar] 3d¹⁰ 4s² 4p²
    50,  # Sn: [Kr] 4d¹⁰ 5s² 5p²
    82,  # Pb: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p²
    114,  # Fl: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p²
    # Group 15 (p³)
    7,  # N: [He] 2s² 2p³ (3 unpaired)
    15,  # P: [Ne] 3s² 3p³
    33,  # As: [Ar] 3d¹⁰ 4s² 4p³
    51,  # Sb: [Kr] 4d¹⁰ 5s² 5p³
    83,  # Bi: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p³
    115,  # Mc: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p³
    # Group 16 (p⁴)
    8,  # O: [He] 2s² 2p⁴ (2 unpaired)
    16,  # S: [Ne] 3s² 3p⁴
    34,  # Se: [Ar] 3d¹⁰ 4s² 4p⁴
    52,  # Te: [Kr] 4d¹⁰ 5s² 5p⁴
    84,  # Po: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁴
    116,  # Lv: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁴
    # Halogens (p⁵)
    9,  # F: [He] 2s² 2p⁵ (1 unpaired)
    17,  # Cl: [Ne] 3s² 3p⁵
    35,  # Br: [Ar] 3d¹⁰ 4s² 4p⁵
    53,  # I: [Kr] 4d¹⁰ 5s² 5p⁵
    85,  # At: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁵
    117,  # Ts: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁵
    # Transition metals - 3d series
    21,  # Sc: [Ar] 3d¹ 4s² (1 unpaired)
    22,  # Ti: [Ar] 3d² 4s² (2 unpaired)
    23,  # V: [Ar] 3d³ 4s² (3 unpaired)
    24,  # Cr: [Ar] 3d⁵ 4s¹ (6 unpaired, empirical)
    25,  # Mn: [Ar] 3d⁵ 4s² (5 unpaired)
    26,  # Fe: [Ar] 3d⁶ 4s² (4 unpaired)
    27,  # Co: [Ar] 3d⁷ 4s² (3 unpaired)
    28,  # Ni: [Ar] 3d⁸ 4s² (2 unpaired)
    29,  # Cu: [Ar] 3d¹⁰ 4s¹ (1 unpaired, empirical)
    # Transition metals - 4d series
    39,  # Y: [Kr] 4d¹ 5s²
    40,  # Zr: [Kr] 4d² 5s²
    41,  # Nb: [Kr] 4d⁴ 5s¹ (empirical)
    42,  # Mo: [Kr] 4d⁵ 5s¹ (empirical)
    43,  # Tc: [Kr] 4d⁵ 5s²
    44,  # Ru: [Kr] 4d⁷ 5s¹ (empirical)
    45,  # Rh: [Kr] 4d⁸ 5s¹ (empirical)
    47,  # Ag: [Kr] 4d¹⁰ 5s¹ (empirical)
    # Transition metals - 5d series
    71,  # Lu: [Xe] 4f¹⁴ 5d¹ 6s²
    72,  # Hf: [Xe] 4f¹⁴ 5d² 6s²
    73,  # Ta: [Xe] 4f¹⁴ 5d³ 6s²
    74,  # W: [Xe] 4f¹⁴ 5d⁴ 6s²
    75,  # Re: [Xe] 4f¹⁴ 5d⁵ 6s²
    76,  # Os: [Xe] 4f¹⁴ 5d⁶ 6s²
    77,  # Ir: [Xe] 4f¹⁴ 5d⁷ 6s²
    78,  # Pt: [Xe] 4f¹⁴ 5d⁹ 6s¹ (empirical)
    79,  # Au: [Xe] 4f¹⁴ 5d¹⁰ 6s¹ (empirical)
    # Transition metals - 6d series
    103,  # Lr: [Rn] 5f¹⁴ 7s² 7p¹
    104,  # Rf: [Rn] 5f¹⁴ 6d² 7s²
    105,  # Db: [Rn] 5f¹⁴ 6d³ 7s²
    106,  # Sg: [Rn] 5f¹⁴ 6d⁴ 7s²
    107,  # Bh: [Rn] 5f¹⁴ 6d⁵ 7s²
    108,  # Hs: [Rn] 5f¹⁴ 6d⁶ 7s²
    109,  # Mt: [Rn] 5f¹⁴ 6d⁷ 7s²
    110,  # Ds: [Rn] 5f¹⁴ 6d⁸ 7s²
    111,  # Rg: [Rn] 5f¹⁴ 6d⁹ 7s²
    # Lanthanides (4f series)
    57,  # La: [Xe] 5d¹ 6s²
    58,  # Ce: [Xe] 4f¹ 5d¹ 6s²
    59,  # Pr: [Xe] 4f³ 6s²
    60,  # Nd: [Xe] 4f⁴ 6s²
    61,  # Pm: [Xe] 4f⁵ 6s²
    62,  # Sm: [Xe] 4f⁶ 6s²
    63,  # Eu: [Xe] 4f⁷ 6s²
    64,  # Gd: [Xe] 4f⁷ 5d¹ 6s²
    65,  # Tb: [Xe] 4f⁹ 6s²
    66,  # Dy: [Xe] 4f¹⁰ 6s²
    67,  # Ho: [Xe] 4f¹¹ 6s²
    68,  # Er: [Xe] 4f¹² 6s²
    69,  # Tm: [Xe] 4f¹³ 6s²
    # Actinides (5f series)
    89,  # Ac: [Rn] 6d¹ 7s²
    90,  # Th: [Rn] 6d² 7s²
    91,  # Pa: [Rn] 5f² 6d¹ 7s²
    92,  # U: [Rn] 5f³ 6d¹ 7s²
    93,  # Np: [Rn] 5f⁴ 6d¹ 7s²
    94,  # Pu: [Rn] 5f⁶ 7s²
    95,  # Am: [Rn] 5f⁷ 7s²
    96,  # Cm: [Rn] 5f⁷ 6d¹ 7s²
    97,  # Bk: [Rn] 5f⁹ 7s²
    98,  # Cf: [Rn] 5f¹⁰ 7s²
    99,  # Es: [Rn] 5f¹¹ 7s²
    100,  # Fm: [Rn] 5f¹² 7s²
    101,  # Md: [Rn] 5f¹³ 7s²
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Conversion factor from Ångströms to Bohr (atomic units of length).
# 1 Å = 1 / a₀ ≈ 1.8897259886 Bohr, where a₀ = 0.529177210903 Å is the
# Bohr radius (NIST 2018 CODATA value).
ANGSTROM_TO_BOHR: float = 1.8897259886
