"""Physical models, systems, and theoretical frameworks for q_block.

This package contains:

- **Data structures**: quantum-mechanical representations of particles and
  orbitals (``Atom``, ``Shell``, ``SubShell``, ``Orbital``, ``SpinOrbital``).
- **Composite systems**: containers that aggregate atoms into physical
  systems (``AtomicSystem``, ``Molecule``, ``Crystal``).
- **Basis functions**: contracted Gaussian-type orbital representations
  (``ContractedGaussianTypeOrbital``, ``CGTOBasis``).
- **Integrals**: overlap, kinetic energy, nuclear attraction, and
  two-electron repulsion integral engines.
- **Initialization**: calculation context classes (``QuantumCalculationContext``,
  ``HartreeFock``, ``RHF``, ``UHF``, ``ROHF``) and nuclear repulsion energy.
"""

from .atom import Atom
from .electron import SpinOrbital, Orbital, SubShell, Shell

# Composite systems (formerly q_block.models)
from .atomic_system import AtomicSystem
from .molecule import Molecule
from .crystal import Crystal

# Basis functions (formerly q_block.models.basis_functions)
from .basis_functions import ContractedGaussianTypeOrbital

# Integrals (formerly q_block.models.integrals)
from .integrals import KineticEnergy, NuclearAttraction, Overlap, TwoElectronRepulsion

# Initialization (formerly q_block.models.initialization)
from .initialization import QuantumCalculationContext, HartreeFock, RHF, UHF, ROHF
from .initialization import NuclearRepulsionEnergy

__all__ = [
    "Atom",
    "AtomicSystem",
    "ContractedGaussianTypeOrbital",
    "Crystal",
    "HartreeFock",
    "KineticEnergy",
    "Molecule",
    "NuclearAttraction",
    "NuclearRepulsionEnergy",
    "Orbital",
    "Overlap",
    "QuantumCalculationContext",
    "RHF",
    "ROHF",
    "Shell",
    "SpinOrbital",
    "SubShell",
    "TwoElectronRepulsion",
    "UHF",
]
