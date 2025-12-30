"""q_block package public surface.

Expose convenience symbols for quick import in tests and examples.

Prefer importing specific symbols from submodules for clarity when
modifying the package internals.
"""

from .atom import Atom, Orbital, Shell, SpinOrbital, SubShell
from .atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z, ATOMS_SYMBOLS_Z_TO_SYMBOL
from .atomic_system import AtomicSystem
from .basis_set_pople import parse_gaussian_basis
from .input_data import InputData
from .coordinates import Coordinates, CartesianCoordinates
from .molecule import Molecule
from .crystal import Crystal

__all__ = [
    "Atom",
    "AtomicSystem",
    "SpinOrbital",
    "Orbital",
    "SubShell",
    "Shell",
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "parse_gaussian_basis",
    "InputData",
    "Coordinates",
    "CartesianCoordinates",
    "Molecule",
    "Crystal",
]
