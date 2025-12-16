"""q_block package public surface.

Expose convenience symbols for quick import in tests and examples.

Prefer importing specific symbols from submodules for clarity when
modifying the package internals.
"""

from .atom import Atom, Orbital, Shell, SpinOrbital, SubShell
from .atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z, ATOMS_SYMBOLS_Z_TO_SYMBOL
from .basis_set_pople import parse_gaussian_basis

__all__ = [
    "Atom",
    "SpinOrbital",
    "Orbital",
    "SubShell",
    "Shell",
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "parse_gaussian_basis",
]
