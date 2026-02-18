"""Integral computation module for quantum-chemistry calculations.

This subpackage provides classes and functions to compute the molecular
integrals required for Hartree-Fock and post-SCF methods:

Overlap Integrals
-----------------
:class:`OverlapMatrix`
    Computes the overlap matrix :math:`S_{\\mu\\nu}` between contracted
    Gaussian basis functions.

Future Extensions
-----------------
- Kinetic energy integrals :math:`T_{\\mu\\nu}`
- Nuclear attraction integrals :math:`V_{\\mu\\nu}`
- Two-electron repulsion integrals :math:`(\\mu\\nu|\\lambda\\sigma)`
"""

from .overlap import OverlapMatrix

__all__ = [
    "OverlapMatrix",
]
