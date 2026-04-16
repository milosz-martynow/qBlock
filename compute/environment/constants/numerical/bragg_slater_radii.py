"""Bragg-Slater atomic radii in Bohr.

This module provides the :data:`BRAGG_SLATER_RADII` dictionary that
maps atomic numbers (:math:`Z`) to covalent radii in Bohr
(:math:`a_0`).  The values are used primarily for scaling the radial
quadrature grid in DFT numerical integration.

Origin and References
---------------------
The concept of "atomic radii" defined by the average distance from the
nucleus within which the bulk of the electron density resides was
introduced by

    W. L. Bragg, "The arrangement of atoms in crystals,"
    *Philosophical Magazine*, **40** (236), 169-189, 1920.

Bragg tabulated radii from X-ray crystallographic measurements of
simple ionic and covalent crystals.

J. C. Slater later extended and systematised these radii across the
periodic table by fitting to self-consistent-field (SCF) calculations:

    J. C. Slater, "Atomic Radii in Crystals,"
    *Journal of Chemical Physics*, **41** (10), 3199-3204, 1964.

The values stored here follow Slater's 1964 tabulation, converted from
Ångström to atomic units (Bohr) using the factor
:math:`1\\,\\text{Å} = 1.8897259886\\,a_0`.

Usage
-----
The dictionary is keyed by atomic number (:math:`Z`).  Elements not
present in the table default to a fallback radius at the call site
(typically 1.5 Bohr).

Units
-----
All values are in **Bohr** (:math:`a_0 \\approx 0.5292\\,\\text{Å}`).
"""

from typing import Dict

BRAGG_SLATER_RADII: Dict[int, float] = {
    1: 0.661,    # H
    2: 0.567,    # He
    3: 2.551,    # Li
    4: 1.984,    # Be
    5: 1.606,    # B
    6: 1.323,    # C
    7: 1.228,    # N
    8: 1.134,    # O
    9: 1.058,    # F
    10: 0.964,   # Ne
    11: 3.402,   # Na
    12: 2.835,   # Mg
    13: 2.268,   # Al
    14: 2.079,   # Si
    15: 1.890,   # P
    16: 1.890,   # S
    17: 1.701,   # Cl
    18: 1.701,   # Ar
    19: 4.158,   # K
    20: 3.402,   # Ca
    21: 3.024,   # Sc
    22: 2.835,   # Ti
    23: 2.646,   # V
    24: 2.646,   # Cr
    25: 2.646,   # Mn
    26: 2.646,   # Fe
    27: 2.457,   # Co
    28: 2.457,   # Ni
    29: 2.457,   # Cu
    30: 2.457,   # Zn
    31: 2.362,   # Ga
    32: 2.268,   # Ge
    33: 2.174,   # As
    34: 2.079,   # Se
    35: 2.079,   # Br
    36: 1.984,   # Kr
}
