"""Constants and lookup tables used across q_block.

This subpackage groups data-only modules (e.g. atomic symbols and numbers).
"""

from .atoms_data import (
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    CLOSED_SHELL_ATOMS,
    OPEN_SHELL_ATOMS,
    ANGSTROM_TO_BOHR,
)

__all__ = [
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "CLOSED_SHELL_ATOMS",
    "OPEN_SHELL_ATOMS",
    "ANGSTROM_TO_BOHR",
]

