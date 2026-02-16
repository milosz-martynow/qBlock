"""q_block package public surface.

Expose convenience symbols for quick import in tests and examples.

Prefer importing specific symbols from submodules for clarity when
modifying the package internals.
"""

from .models.atom import Atom, Orbital, Shell, SpinOrbital, SubShell
from .constants.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z, ATOMS_SYMBOLS_Z_TO_SYMBOL
from .systems.atomic_system import AtomicSystem
from .io.basis_set import BasisSet, Pople
from .io.input_data import InputData
from .io.coordinates import Coordinates, CartesianCoordinates
from .systems.molecule import Molecule
from .systems.crystal import Crystal

# Theory module - initialization for quantum-chemistry calculations
from .theory.initialization import Initialization, HartreeFock, RHF, UHF, ROHF
from .theory.basis_functions import ContractedGaussianTypeOrbital
from .constants.atoms_data import ANGSTROM_TO_BOHR

__all__ = [
    "Atom",
    "AtomicSystem",
    "SpinOrbital",
    "Orbital",
    "SubShell",
    "Shell",
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "ANGSTROM_TO_BOHR",
    "BasisSet",
    "Pople",
    "InputData",
    "Coordinates",
    "CartesianCoordinates",
    "Molecule",
    "Crystal",
    "Initialization",
    "HartreeFock",
    "RHF",
    "UHF",
    "ROHF",
    "ContractedGaussianTypeOrbital",
]
