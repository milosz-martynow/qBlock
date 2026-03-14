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
        {"n": 4, "l": 0, "electron_count": 1},   # 4s^1
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
        {"n": 5, "l": 0, "electron_count": 1},   # 5s^1
    ],
    # Platinum: [Xe] 4f^14 5d^9 6s^1 instead of [Xe] 4f^14 5d^8 6s^2
    78: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f^14
        {"n": 5, "l": 2, "electron_count": 9},   # 5d^9
        {"n": 6, "l": 0, "electron_count": 1},   # 6s^1
    ],
    # Gold: [Xe] 4f^14 5d^10 6s^1 instead of [Xe] 4f^14 5d^9 6s^2
    79: [
        {"n": 4, "l": 3, "electron_count": 14},  # 4f^14
        {"n": 5, "l": 2, "electron_count": 10},  # 5d^10
        {"n": 6, "l": 0, "electron_count": 1},   # 6s^1
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
    2,    # He: 1s²
    10,   # Ne: [He] 2s² 2p⁶
    18,   # Ar: [Ne] 3s² 3p⁶
    36,   # Kr: [Ar] 3d¹⁰ 4s² 4p⁶
    54,   # Xe: [Kr] 4d¹⁰ 5s² 5p⁶
    86,   # Rn: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁶
    118,  # Og: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁶
    
    # Alkaline earth metals (ns²)
    4,    # Be: [He] 2s²
    12,   # Mg: [Ne] 3s²
    20,   # Ca: [Ar] 4s²
    38,   # Sr: [Kr] 5s²
    56,   # Ba: [Xe] 6s²
    88,   # Ra: [Rn] 7s²
    
    # Group 12 (d¹⁰ s²)
    30,   # Zn: [Ar] 3d¹⁰ 4s²
    48,   # Cd: [Kr] 4d¹⁰ 5s²
    80,   # Hg: [Xe] 4f¹⁴ 5d¹⁰ 6s²
    112,  # Cn: [Rn] 5f¹⁴ 6d¹⁰ 7s²
    
    # Palladium (empirical exception: 4d¹⁰, no 5s electrons)
    46,   # Pd: [Kr] 4d¹⁰
    
    # Ytterbium (f¹⁴ s²)
    70,   # Yb: [Xe] 4f¹⁴ 6s²
    
    # Nobelium (f¹⁴ s²)
    102,  # No: [Rn] 5f¹⁴ 7s²
]

# Open-shell atoms: have at least one unpaired electron
# Includes all atoms not in CLOSED_SHELL_ATOMS
OPEN_SHELL_ATOMS: List[int] = [
    # Hydrogen
    1,    # H: 1s¹ (1 unpaired)
    
    # Alkali metals (ns¹)
    3,    # Li: [He] 2s¹
    11,   # Na: [Ne] 3s¹
    19,   # K: [Ar] 4s¹
    37,   # Rb: [Kr] 5s¹
    55,   # Cs: [Xe] 6s¹
    87,   # Fr: [Rn] 7s¹
    
    # Group 13 (p¹)
    5,    # B: [He] 2s² 2p¹
    13,   # Al: [Ne] 3s² 3p¹
    31,   # Ga: [Ar] 3d¹⁰ 4s² 4p¹
    49,   # In: [Kr] 4d¹⁰ 5s² 5p¹
    81,   # Tl: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p¹
    113,  # Nh: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p¹
    
    # Group 14 (p²)
    6,    # C: [He] 2s² 2p² (2 unpaired)
    14,   # Si: [Ne] 3s² 3p²
    32,   # Ge: [Ar] 3d¹⁰ 4s² 4p²
    50,   # Sn: [Kr] 4d¹⁰ 5s² 5p²
    82,   # Pb: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p²
    114,  # Fl: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p²
    
    # Group 15 (p³)
    7,    # N: [He] 2s² 2p³ (3 unpaired)
    15,   # P: [Ne] 3s² 3p³
    33,   # As: [Ar] 3d¹⁰ 4s² 4p³
    51,   # Sb: [Kr] 4d¹⁰ 5s² 5p³
    83,   # Bi: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p³
    115,  # Mc: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p³
    
    # Group 16 (p⁴)
    8,    # O: [He] 2s² 2p⁴ (2 unpaired)
    16,   # S: [Ne] 3s² 3p⁴
    34,   # Se: [Ar] 3d¹⁰ 4s² 4p⁴
    52,   # Te: [Kr] 4d¹⁰ 5s² 5p⁴
    84,   # Po: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁴
    116,  # Lv: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁴
    
    # Halogens (p⁵)
    9,    # F: [He] 2s² 2p⁵ (1 unpaired)
    17,   # Cl: [Ne] 3s² 3p⁵
    35,   # Br: [Ar] 3d¹⁰ 4s² 4p⁵
    53,   # I: [Kr] 4d¹⁰ 5s² 5p⁵
    85,   # At: [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁵
    117,  # Ts: [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁵
    
    # Transition metals - 3d series
    21,   # Sc: [Ar] 3d¹ 4s² (1 unpaired)
    22,   # Ti: [Ar] 3d² 4s² (2 unpaired)
    23,   # V: [Ar] 3d³ 4s² (3 unpaired)
    24,   # Cr: [Ar] 3d⁵ 4s¹ (6 unpaired, empirical)
    25,   # Mn: [Ar] 3d⁵ 4s² (5 unpaired)
    26,   # Fe: [Ar] 3d⁶ 4s² (4 unpaired)
    27,   # Co: [Ar] 3d⁷ 4s² (3 unpaired)
    28,   # Ni: [Ar] 3d⁸ 4s² (2 unpaired)
    29,   # Cu: [Ar] 3d¹⁰ 4s¹ (1 unpaired, empirical)
    
    # Transition metals - 4d series
    39,   # Y: [Kr] 4d¹ 5s²
    40,   # Zr: [Kr] 4d² 5s²
    41,   # Nb: [Kr] 4d⁴ 5s¹ (empirical)
    42,   # Mo: [Kr] 4d⁵ 5s¹ (empirical)
    43,   # Tc: [Kr] 4d⁵ 5s²
    44,   # Ru: [Kr] 4d⁷ 5s¹ (empirical)
    45,   # Rh: [Kr] 4d⁸ 5s¹ (empirical)
    47,   # Ag: [Kr] 4d¹⁰ 5s¹ (empirical)
    
    # Transition metals - 5d series
    71,   # Lu: [Xe] 4f¹⁴ 5d¹ 6s²
    72,   # Hf: [Xe] 4f¹⁴ 5d² 6s²
    73,   # Ta: [Xe] 4f¹⁴ 5d³ 6s²
    74,   # W: [Xe] 4f¹⁴ 5d⁴ 6s²
    75,   # Re: [Xe] 4f¹⁴ 5d⁵ 6s²
    76,   # Os: [Xe] 4f¹⁴ 5d⁶ 6s²
    77,   # Ir: [Xe] 4f¹⁴ 5d⁷ 6s²
    78,   # Pt: [Xe] 4f¹⁴ 5d⁹ 6s¹ (empirical)
    79,   # Au: [Xe] 4f¹⁴ 5d¹⁰ 6s¹ (empirical)
    
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
    57,   # La: [Xe] 5d¹ 6s²
    58,   # Ce: [Xe] 4f¹ 5d¹ 6s²
    59,   # Pr: [Xe] 4f³ 6s²
    60,   # Nd: [Xe] 4f⁴ 6s²
    61,   # Pm: [Xe] 4f⁵ 6s²
    62,   # Sm: [Xe] 4f⁶ 6s²
    63,   # Eu: [Xe] 4f⁷ 6s²
    64,   # Gd: [Xe] 4f⁷ 5d¹ 6s²
    65,   # Tb: [Xe] 4f⁹ 6s²
    66,   # Dy: [Xe] 4f¹⁰ 6s²
    67,   # Ho: [Xe] 4f¹¹ 6s²
    68,   # Er: [Xe] 4f¹² 6s²
    69,   # Tm: [Xe] 4f¹³ 6s²
    
    # Actinides (5f series)
    89,   # Ac: [Rn] 6d¹ 7s²
    90,   # Th: [Rn] 6d² 7s²
    91,   # Pa: [Rn] 5f² 6d¹ 7s²
    92,   # U: [Rn] 5f³ 6d¹ 7s²
    93,   # Np: [Rn] 5f⁴ 6d¹ 7s²
    94,   # Pu: [Rn] 5f⁶ 7s²
    95,   # Am: [Rn] 5f⁷ 7s²
    96,   # Cm: [Rn] 5f⁷ 6d¹ 7s²
    97,   # Bk: [Rn] 5f⁹ 7s²
    98,   # Cf: [Rn] 5f¹⁰ 7s²
    99,   # Es: [Rn] 5f¹¹ 7s²
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

# ---------------------------------------------------------------------------
# Reference data: Atomic ionization energies for Koopmans' theorem testing
# ---------------------------------------------------------------------------
#
# Dictionary mapping atomic number Z (1–54) to reference data for testing
# Hartree-Fock orbital energies via Koopmans' theorem:  IE ≈ −ε_HOMO.
#
# Fields per entry:
#   symbol              – Element symbol
#   config              – Ground-state electronic configuration
#   multiplicity        – Spin multiplicity 2S+1
#   n_electrons         – Total electron count (= Z for neutral atoms)
#   n_alpha, n_beta     – UHF spin-channel occupations
#   n_closed, n_open    – ROHF doubly / singly occupied spatial orbital counts
#   experimental_ie_eV  – Experimental first ionization energy in eV
#   hf_ie_eV            – Approximate Koopmans' theorem IE at
#                         the numerical Hartree-Fock limit in eV
#
# Sources (openly accessible):
#   Experimental IE:
#       Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2023).
#       NIST Atomic Spectra Database (ver. 5.11).
#       National Institute of Standards and Technology, Gaithersburg, MD.
#       https://physics.nist.gov/asd
#   HF theoretical IE:
#       NIST Computational Chemistry Comparison and Benchmark Database,
#       NIST Standard Reference Database Number 101, Release 22, May 2022,
#       Editor: Russell D. Johnson III.
#       https://cccbdb.nist.gov/   DOI:10.18434/T47C7Z
#
# Notes:
#   - HF IE values are approximate.  Actual computed values depend on
#     the basis set used.  The numbers below correspond to near-basis-set-
#     limit Hartree-Fock calculations.
#   - For closed-shell atoms (multiplicity == 1), use RHF.
#   - For open-shell atoms  (multiplicity >  1), use UHF or ROHF.
#   - Conversion factor: 1 Hartree = 27.211386 eV.
# ---------------------------------------------------------------------------

ATOMS_HOMO_ENERGIES: dict = {
    # ---- Period 1 --------------------------------------------------------
    1: {
        "symbol": "H", "config": "1s1", "multiplicity": 2,
        "n_electrons": 1, "n_alpha": 1, "n_beta": 0,
        "n_closed": 0, "n_open": 1,
        "experimental_ie_eV": 13.5984, "hf_ie_eV": 13.61,
    },
    2: {
        "symbol": "He", "config": "1s2", "multiplicity": 1,
        "n_electrons": 2, "n_alpha": 1, "n_beta": 1,
        "n_closed": 1, "n_open": 0,
        "experimental_ie_eV": 24.5874, "hf_ie_eV": 24.98,
    },
    # ---- Period 2 --------------------------------------------------------
    3: {
        "symbol": "Li", "config": "[He] 2s1", "multiplicity": 2,
        "n_electrons": 3, "n_alpha": 2, "n_beta": 1,
        "n_closed": 1, "n_open": 1,
        "experimental_ie_eV": 5.3917, "hf_ie_eV": 5.34,
    },
    4: {
        "symbol": "Be", "config": "[He] 2s2", "multiplicity": 1,
        "n_electrons": 4, "n_alpha": 2, "n_beta": 2,
        "n_closed": 2, "n_open": 0,
        "experimental_ie_eV": 9.3227, "hf_ie_eV": 8.42,
    },
    5: {
        "symbol": "B", "config": "[He] 2s2 2p1", "multiplicity": 2,
        "n_electrons": 5, "n_alpha": 3, "n_beta": 2,
        "n_closed": 2, "n_open": 1,
        "experimental_ie_eV": 8.2980, "hf_ie_eV": 8.43,
    },
    6: {
        "symbol": "C", "config": "[He] 2s2 2p2", "multiplicity": 3,
        "n_electrons": 6, "n_alpha": 4, "n_beta": 2,
        "n_closed": 2, "n_open": 2,
        "experimental_ie_eV": 11.2603, "hf_ie_eV": 11.79,
    },
    7: {
        "symbol": "N", "config": "[He] 2s2 2p3", "multiplicity": 4,
        "n_electrons": 7, "n_alpha": 5, "n_beta": 2,
        "n_closed": 2, "n_open": 3,
        "experimental_ie_eV": 14.5341, "hf_ie_eV": 15.44,
    },
    8: {
        "symbol": "O", "config": "[He] 2s2 2p4", "multiplicity": 3,
        "n_electrons": 8, "n_alpha": 5, "n_beta": 3,
        "n_closed": 3, "n_open": 2,
        "experimental_ie_eV": 13.6181, "hf_ie_eV": 12.44,
    },
    9: {
        "symbol": "F", "config": "[He] 2s2 2p5", "multiplicity": 2,
        "n_electrons": 9, "n_alpha": 5, "n_beta": 4,
        "n_closed": 4, "n_open": 1,
        "experimental_ie_eV": 17.4228, "hf_ie_eV": 16.22,
    },
    10: {
        "symbol": "Ne", "config": "[He] 2s2 2p6", "multiplicity": 1,
        "n_electrons": 10, "n_alpha": 5, "n_beta": 5,
        "n_closed": 5, "n_open": 0,
        "experimental_ie_eV": 21.5645, "hf_ie_eV": 23.14,
    },
    # ---- Period 3 --------------------------------------------------------
    11: {
        "symbol": "Na", "config": "[Ne] 3s1", "multiplicity": 2,
        "n_electrons": 11, "n_alpha": 6, "n_beta": 5,
        "n_closed": 5, "n_open": 1,
        "experimental_ie_eV": 5.1391, "hf_ie_eV": 5.02,
    },
    12: {
        "symbol": "Mg", "config": "[Ne] 3s2", "multiplicity": 1,
        "n_electrons": 12, "n_alpha": 6, "n_beta": 6,
        "n_closed": 6, "n_open": 0,
        "experimental_ie_eV": 7.6462, "hf_ie_eV": 6.89,
    },
    13: {
        "symbol": "Al", "config": "[Ne] 3s2 3p1", "multiplicity": 2,
        "n_electrons": 13, "n_alpha": 7, "n_beta": 6,
        "n_closed": 6, "n_open": 1,
        "experimental_ie_eV": 5.9858, "hf_ie_eV": 5.71,
    },
    14: {
        "symbol": "Si", "config": "[Ne] 3s2 3p2", "multiplicity": 3,
        "n_electrons": 14, "n_alpha": 8, "n_beta": 6,
        "n_closed": 6, "n_open": 2,
        "experimental_ie_eV": 8.1517, "hf_ie_eV": 8.14,
    },
    15: {
        "symbol": "P", "config": "[Ne] 3s2 3p3", "multiplicity": 4,
        "n_electrons": 15, "n_alpha": 9, "n_beta": 6,
        "n_closed": 6, "n_open": 3,
        "experimental_ie_eV": 10.4867, "hf_ie_eV": 10.67,
    },
    16: {
        "symbol": "S", "config": "[Ne] 3s2 3p4", "multiplicity": 3,
        "n_electrons": 16, "n_alpha": 9, "n_beta": 7,
        "n_closed": 7, "n_open": 2,
        "experimental_ie_eV": 10.3600, "hf_ie_eV": 9.63,
    },
    17: {
        "symbol": "Cl", "config": "[Ne] 3s2 3p5", "multiplicity": 2,
        "n_electrons": 17, "n_alpha": 9, "n_beta": 8,
        "n_closed": 8, "n_open": 1,
        "experimental_ie_eV": 12.9676, "hf_ie_eV": 12.42,
    },
    18: {
        "symbol": "Ar", "config": "[Ne] 3s2 3p6", "multiplicity": 1,
        "n_electrons": 18, "n_alpha": 9, "n_beta": 9,
        "n_closed": 9, "n_open": 0,
        "experimental_ie_eV": 15.7596, "hf_ie_eV": 16.08,
    },
    # ---- Period 4 --------------------------------------------------------
    19: {
        "symbol": "K", "config": "[Ar] 4s1", "multiplicity": 2,
        "n_electrons": 19, "n_alpha": 10, "n_beta": 9,
        "n_closed": 9, "n_open": 1,
        "experimental_ie_eV": 4.3407, "hf_ie_eV": 4.19,
    },
    20: {
        "symbol": "Ca", "config": "[Ar] 4s2", "multiplicity": 1,
        "n_electrons": 20, "n_alpha": 10, "n_beta": 10,
        "n_closed": 10, "n_open": 0,
        "experimental_ie_eV": 6.1132, "hf_ie_eV": 5.49,
    },
    21: {
        "symbol": "Sc", "config": "[Ar] 3d1 4s2", "multiplicity": 2,
        "n_electrons": 21, "n_alpha": 11, "n_beta": 10,
        "n_closed": 10, "n_open": 1,
        "experimental_ie_eV": 6.5615, "hf_ie_eV": 5.67,
    },
    22: {
        "symbol": "Ti", "config": "[Ar] 3d2 4s2", "multiplicity": 3,
        "n_electrons": 22, "n_alpha": 12, "n_beta": 10,
        "n_closed": 10, "n_open": 2,
        "experimental_ie_eV": 6.8281, "hf_ie_eV": 5.98,
    },
    23: {
        "symbol": "V", "config": "[Ar] 3d3 4s2", "multiplicity": 4,
        "n_electrons": 23, "n_alpha": 13, "n_beta": 10,
        "n_closed": 10, "n_open": 3,
        "experimental_ie_eV": 6.7462, "hf_ie_eV": 5.92,
    },
    24: {
        "symbol": "Cr", "config": "[Ar] 3d5 4s1", "multiplicity": 7,
        "n_electrons": 24, "n_alpha": 15, "n_beta": 9,
        "n_closed": 9, "n_open": 6,
        "experimental_ie_eV": 6.7665, "hf_ie_eV": 5.79,
    },
    25: {
        "symbol": "Mn", "config": "[Ar] 3d5 4s2", "multiplicity": 6,
        "n_electrons": 25, "n_alpha": 15, "n_beta": 10,
        "n_closed": 10, "n_open": 5,
        "experimental_ie_eV": 7.4340, "hf_ie_eV": 6.66,
    },
    26: {
        "symbol": "Fe", "config": "[Ar] 3d6 4s2", "multiplicity": 5,
        "n_electrons": 26, "n_alpha": 15, "n_beta": 11,
        "n_closed": 11, "n_open": 4,
        "experimental_ie_eV": 7.9024, "hf_ie_eV": 6.65,
    },
    27: {
        "symbol": "Co", "config": "[Ar] 3d7 4s2", "multiplicity": 4,
        "n_electrons": 27, "n_alpha": 15, "n_beta": 12,
        "n_closed": 12, "n_open": 3,
        "experimental_ie_eV": 7.8810, "hf_ie_eV": 6.50,
    },
    28: {
        "symbol": "Ni", "config": "[Ar] 3d8 4s2", "multiplicity": 3,
        "n_electrons": 28, "n_alpha": 15, "n_beta": 13,
        "n_closed": 13, "n_open": 2,
        "experimental_ie_eV": 7.6398, "hf_ie_eV": 6.42,
    },
    29: {
        "symbol": "Cu", "config": "[Ar] 3d10 4s1", "multiplicity": 2,
        "n_electrons": 29, "n_alpha": 15, "n_beta": 14,
        "n_closed": 14, "n_open": 1,
        "experimental_ie_eV": 7.7264, "hf_ie_eV": 6.49,
    },
    30: {
        "symbol": "Zn", "config": "[Ar] 3d10 4s2", "multiplicity": 1,
        "n_electrons": 30, "n_alpha": 15, "n_beta": 15,
        "n_closed": 15, "n_open": 0,
        "experimental_ie_eV": 9.3942, "hf_ie_eV": 8.61,
    },
    31: {
        "symbol": "Ga", "config": "[Ar] 3d10 4s2 4p1", "multiplicity": 2,
        "n_electrons": 31, "n_alpha": 16, "n_beta": 15,
        "n_closed": 15, "n_open": 1,
        "experimental_ie_eV": 5.9993, "hf_ie_eV": 5.59,
    },
    32: {
        "symbol": "Ge", "config": "[Ar] 3d10 4s2 4p2", "multiplicity": 3,
        "n_electrons": 32, "n_alpha": 17, "n_beta": 15,
        "n_closed": 15, "n_open": 2,
        "experimental_ie_eV": 7.8994, "hf_ie_eV": 7.76,
    },
    33: {
        "symbol": "As", "config": "[Ar] 3d10 4s2 4p3", "multiplicity": 4,
        "n_electrons": 33, "n_alpha": 18, "n_beta": 15,
        "n_closed": 15, "n_open": 3,
        "experimental_ie_eV": 9.7886, "hf_ie_eV": 9.95,
    },
    34: {
        "symbol": "Se", "config": "[Ar] 3d10 4s2 4p4", "multiplicity": 3,
        "n_electrons": 34, "n_alpha": 18, "n_beta": 16,
        "n_closed": 16, "n_open": 2,
        "experimental_ie_eV": 9.7524, "hf_ie_eV": 8.99,
    },
    35: {
        "symbol": "Br", "config": "[Ar] 3d10 4s2 4p5", "multiplicity": 2,
        "n_electrons": 35, "n_alpha": 18, "n_beta": 17,
        "n_closed": 17, "n_open": 1,
        "experimental_ie_eV": 11.8138, "hf_ie_eV": 11.23,
    },
    36: {
        "symbol": "Kr", "config": "[Ar] 3d10 4s2 4p6", "multiplicity": 1,
        "n_electrons": 36, "n_alpha": 18, "n_beta": 18,
        "n_closed": 18, "n_open": 0,
        "experimental_ie_eV": 13.9996, "hf_ie_eV": 14.26,
    },
    # ---- Period 5 --------------------------------------------------------
    37: {
        "symbol": "Rb", "config": "[Kr] 5s1", "multiplicity": 2,
        "n_electrons": 37, "n_alpha": 19, "n_beta": 18,
        "n_closed": 18, "n_open": 1,
        "experimental_ie_eV": 4.1771, "hf_ie_eV": 3.98,
    },
    38: {
        "symbol": "Sr", "config": "[Kr] 5s2", "multiplicity": 1,
        "n_electrons": 38, "n_alpha": 19, "n_beta": 19,
        "n_closed": 19, "n_open": 0,
        "experimental_ie_eV": 5.6949, "hf_ie_eV": 5.03,
    },
    39: {
        "symbol": "Y", "config": "[Kr] 4d1 5s2", "multiplicity": 2,
        "n_electrons": 39, "n_alpha": 20, "n_beta": 19,
        "n_closed": 19, "n_open": 1,
        "experimental_ie_eV": 6.2173, "hf_ie_eV": 5.39,
    },
    40: {
        "symbol": "Zr", "config": "[Kr] 4d2 5s2", "multiplicity": 3,
        "n_electrons": 40, "n_alpha": 21, "n_beta": 19,
        "n_closed": 19, "n_open": 2,
        "experimental_ie_eV": 6.6339, "hf_ie_eV": 5.70,
    },
    41: {
        "symbol": "Nb", "config": "[Kr] 4d4 5s1", "multiplicity": 6,
        "n_electrons": 41, "n_alpha": 23, "n_beta": 18,
        "n_closed": 18, "n_open": 5,
        "experimental_ie_eV": 6.7589, "hf_ie_eV": 5.82,
    },
    42: {
        "symbol": "Mo", "config": "[Kr] 4d5 5s1", "multiplicity": 7,
        "n_electrons": 42, "n_alpha": 24, "n_beta": 18,
        "n_closed": 18, "n_open": 6,
        "experimental_ie_eV": 7.0924, "hf_ie_eV": 6.02,
    },
    43: {
        "symbol": "Tc", "config": "[Kr] 4d5 5s2", "multiplicity": 6,
        "n_electrons": 43, "n_alpha": 24, "n_beta": 19,
        "n_closed": 19, "n_open": 5,
        "experimental_ie_eV": 7.1190, "hf_ie_eV": 6.37,
    },
    44: {
        "symbol": "Ru", "config": "[Kr] 4d7 5s1", "multiplicity": 5,
        "n_electrons": 44, "n_alpha": 24, "n_beta": 20,
        "n_closed": 20, "n_open": 4,
        "experimental_ie_eV": 7.3605, "hf_ie_eV": 6.33,
    },
    45: {
        "symbol": "Rh", "config": "[Kr] 4d8 5s1", "multiplicity": 4,
        "n_electrons": 45, "n_alpha": 24, "n_beta": 21,
        "n_closed": 21, "n_open": 3,
        "experimental_ie_eV": 7.4589, "hf_ie_eV": 6.30,
    },
    46: {
        "symbol": "Pd", "config": "[Kr] 4d10", "multiplicity": 1,
        "n_electrons": 46, "n_alpha": 23, "n_beta": 23,
        "n_closed": 23, "n_open": 0,
        "experimental_ie_eV": 8.3369, "hf_ie_eV": 7.40,
    },
    47: {
        "symbol": "Ag", "config": "[Kr] 4d10 5s1", "multiplicity": 2,
        "n_electrons": 47, "n_alpha": 24, "n_beta": 23,
        "n_closed": 23, "n_open": 1,
        "experimental_ie_eV": 7.5762, "hf_ie_eV": 6.25,
    },
    48: {
        "symbol": "Cd", "config": "[Kr] 4d10 5s2", "multiplicity": 1,
        "n_electrons": 48, "n_alpha": 24, "n_beta": 24,
        "n_closed": 24, "n_open": 0,
        "experimental_ie_eV": 8.9938, "hf_ie_eV": 8.15,
    },
    49: {
        "symbol": "In", "config": "[Kr] 4d10 5s2 5p1", "multiplicity": 2,
        "n_electrons": 49, "n_alpha": 25, "n_beta": 24,
        "n_closed": 24, "n_open": 1,
        "experimental_ie_eV": 5.7864, "hf_ie_eV": 5.27,
    },
    50: {
        "symbol": "Sn", "config": "[Kr] 4d10 5s2 5p2", "multiplicity": 3,
        "n_electrons": 50, "n_alpha": 26, "n_beta": 24,
        "n_closed": 24, "n_open": 2,
        "experimental_ie_eV": 7.3439, "hf_ie_eV": 7.16,
    },
    51: {
        "symbol": "Sb", "config": "[Kr] 4d10 5s2 5p3", "multiplicity": 4,
        "n_electrons": 51, "n_alpha": 27, "n_beta": 24,
        "n_closed": 24, "n_open": 3,
        "experimental_ie_eV": 8.6084, "hf_ie_eV": 8.76,
    },
    52: {
        "symbol": "Te", "config": "[Kr] 4d10 5s2 5p4", "multiplicity": 3,
        "n_electrons": 52, "n_alpha": 27, "n_beta": 25,
        "n_closed": 25, "n_open": 2,
        "experimental_ie_eV": 9.0096, "hf_ie_eV": 8.24,
    },
    53: {
        "symbol": "I", "config": "[Kr] 4d10 5s2 5p5", "multiplicity": 2,
        "n_electrons": 53, "n_alpha": 27, "n_beta": 26,
        "n_closed": 26, "n_open": 1,
        "experimental_ie_eV": 10.4513, "hf_ie_eV": 10.07,
    },
    54: {
        "symbol": "Xe", "config": "[Kr] 4d10 5s2 5p6", "multiplicity": 1,
        "n_electrons": 54, "n_alpha": 27, "n_beta": 27,
        "n_closed": 27, "n_open": 0,
        "experimental_ie_eV": 12.1298, "hf_ie_eV": 12.44,
    },
}
