"""Constants and lookup tables.

Subpackages
-----------
natural
    Physical constants and atomic data derived from nature
    (element symbols, angular momentum mappings, unit conversions).
numerical
    Numerical data sets used in calculations (basis set files,
    pseudopotential parameters, etc.).
"""

from .natural.atoms_data import (
    ANGSTROM_TO_BOHR,
    ANGULAR_MOMENTUM_MAP,
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    CLOSED_SHELL_ATOMS,
    EMPIRICAL_EXCEPTIONS,
    OPEN_SHELL_ATOMS,
)

__all__ = [
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "CLOSED_SHELL_ATOMS",
    "OPEN_SHELL_ATOMS",
    "ANGSTROM_TO_BOHR",
]
