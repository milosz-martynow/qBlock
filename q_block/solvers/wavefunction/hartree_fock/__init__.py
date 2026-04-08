"""
Hartree-Fock methods.

This module contains implementations of Hartree-Fock theory, which approximates
the many-electron wavefunction as a single Slater determinant and optimizes
the molecular orbitals via the self-consistent field (SCF) procedure.

Classes:
    HartreeFock: Abstract base class for all HF variants.
    RestrictedHartreeFock: Restricted HF (closed-shell systems).
    UnrestrictedHartreeFock: Unrestricted HF (open-shell, different α/β orbitals).
    RestrictedOpenShellHartreeFock: Restricted open-shell HF (shared spatial MOs).
"""

from ...spin_pair import SpinPair
from .hartree_fock import HartreeFock
from .restricted_hartree_fock import RestrictedHartreeFock
from .unrestricted_hartree_fock import UnrestrictedHartreeFock
from .restricted_open_shell_hartree_fock import (
    RestrictedOpenShellHartreeFock,
)

__all__ = [
    "HartreeFock",
    "SpinPair",
    "RestrictedHartreeFock",
    "UnrestrictedHartreeFock",
    "RestrictedOpenShellHartreeFock",
]
