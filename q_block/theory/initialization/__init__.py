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
    Common Hartree-Fock base class: inherits the generic class and adds
    basis-set validation and shared electron-count attributes.

RHF
    Restricted Closed-Shell Hartree-Fock (singlet, even electrons).

UHF
    Unrestricted Hartree-Fock (any multiplicity).

ROHF
    Restricted Open-Shell Hartree-Fock.
"""

from .initialization import Initialization, HartreeFock, RHF, UHF, ROHF

__all__ = [
    "Initialization",
    "HartreeFock",
    "RHF",
    "UHF",
    "ROHF",
]
