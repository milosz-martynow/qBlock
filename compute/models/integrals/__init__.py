"""Integral computation module.

This module provides classes for computing integral matrices:

- :class:`Integral`: Abstract base class for all integral matrices
- :class:`TwoGaussianIntegral`: Base class for Gaussian-type orbital integrals
- :class:`Overlap`: Overlap matrix S
- :class:`KineticEnergy`: Kinetic energy matrix T
- :class:`NuclearAttraction`: Nuclear attraction matrix V
- :class:`TwoElectronRepulsion`: Two-electron repulsion integral tensor (μν|λσ)
"""

from .integral import Integral
from .kinetic_energy import KineticEnergy
from .nuclear_attraction import NuclearAttraction
from .overlap import Overlap
from .two_electron_repulsion import TwoElectronRepulsion
from .two_gaussian_integral import TwoGaussianIntegral

__all__ = [
    "Integral",
    "KineticEnergy",
    "NuclearAttraction",
    "Overlap",
    "TwoElectronRepulsion",
    "TwoGaussianIntegral",
]
