"""Integral computation module.

This module provides classes for computing one-electron integral matrices:

- :class:`Integral`: Abstract base class for all integral matrices
- :class:`TwoGaussianIntegral`: Base class for Gaussian-type orbital integrals
- :class:`Overlap`: Overlap matrix S
- :class:`KineticEnergy`: Kinetic energy matrix T
- :class:`NuclearAttraction`: Nuclear attraction matrix V
"""

from .integral import Integral
from .kinetic_energy import KineticEnergy
from .nuclear_attraction import NuclearAttraction
from .overlap import Overlap
from .two_gaussian_integral import TwoGaussianIntegral

__all__ = [
    "Integral",
    "KineticEnergy",
    "NuclearAttraction",
    "Overlap",
    "TwoGaussianIntegral",
]
