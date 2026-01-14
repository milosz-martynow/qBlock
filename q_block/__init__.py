"""q_block package public surface.

Expose convenience symbols for quick import in tests and examples.

Prefer importing specific symbols from submodules for clarity when
modifying the package internals.
"""

from .models.atom import Atom, Orbital, Shell, SpinOrbital, SubShell
from .constants.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z, ATOMS_SYMBOLS_Z_TO_SYMBOL
from .systems.atomic_system import AtomicSystem
from .io.basis_set_pople import parse_gaussian_basis
from .io.input_data import InputData
from .io.coordinates import Coordinates, CartesianCoordinates
from .systems.molecule import Molecule
from .systems.crystal import Crystal

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
