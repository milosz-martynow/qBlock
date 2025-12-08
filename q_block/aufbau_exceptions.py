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
https://physics.nist.gov/PhysRefData/ASD/
"""

from typing import Dict, List

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
