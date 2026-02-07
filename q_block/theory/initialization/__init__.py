"""Initialization helpers for quantum-chemistry calculations.

This subpackage provides base and method-specific initialization classes
that load molecular geometry, validate input parameters, and prepare
data structures required before any SCF or post-SCF computation.

Classes
-------
Initialization
    Generic (parent) initialization: loads geometry, charge, multiplicity,
    converts coordinates to Bohr, computes the total electron count.

HartreeFock
    Hartree-Fock-specific initialization: inherits the generic class and
    adds HF method selection (RHF/UHF/ROHF), basis-set validation, and
    electron-count bookkeeping appropriate for each HF variant.
"""

from .initialization import Initialization, HartreeFock

__all__ = [
    "Initialization",
    "HartreeFock",
]
