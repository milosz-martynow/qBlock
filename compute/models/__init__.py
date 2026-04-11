"""Physical models, systems, and theoretical frameworks for compute.

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

# Composite systems (formerly compute.models)
from .atomic_system import AtomicSystem

# Basis functions (formerly compute.models.basis_functions)
from .basis_functions import ContractedGaussianTypeOrbital
from .crystal import Crystal
from .electron import Orbital, Shell, SpinOrbital, SubShell

# Initialization (formerly compute.models.initialization)
from .initialization import (
    RHF,
    ROHF,
    UHF,
    HartreeFock,
    NuclearRepulsionEnergy,
    QuantumCalculationContext,
)

# Integrals (formerly compute.models.integrals)
from .integrals import (
    KineticEnergy,
    NuclearAttraction,
    Overlap,
    TwoElectronRepulsion,
)
from .molecule import Molecule

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
