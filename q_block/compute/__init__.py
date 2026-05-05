r"""compute — a modular quantum-chemistry framework.

Package structure
=================

q_block/compute/
├── environment/        Foundational infrastructure and external data.
│   │                   Consolidates all external data sources,
│   │                   configuration, and I/O interfaces.
│   │
│   ├── constants/      Domain-specific constants and reference data.
│   │   ├── natural/    Physical constants and atomic data derived from
│   │   │               nature (element symbols, angular momentum maps,
│   │   │               unit conversion factors, empirical electron
│   │   │               configuration exceptions).
│   │   └── numerical/  Numerical data sets used in calculations
│   │                   (basis set parameter files such as Pople .gbs).
│   │
│   ├── io/             Input / output layer.
│   │                   Parsers and serialisers for user-facing data
│   │                   formats: basis-set files (``BasisSet``, ``Pople``),
│   │                   Cartesian coordinates (``Coordinates``,
│   │                   ``CartesianCoordinates``), composite input
│   │                   containers (``InputData``), and output containers
│   │                   (``OutputData``).
│   │
│   └── configuration.py  Input configuration reader for plain-text
│                          ``.qblock.config`` files
│                          (``Configuration``).
│
│   └── logs.py            Centralised logging configuration
│                          (``setup_logging``).
│
├── models/             Physical models, systems, and theory.
│   │                   Everything that mathematically describes the
│   │                   physical world lives here — data structures for
│   │                   particles and orbitals, composite systems, basis
│   │                   function representations, and integral engines.
│   │
│   │  Data structures
│   ├── atom.py             Atom (shells → subshells → orbitals → spin-orbitals)
│   ├── electron.py         SpinOrbital, Orbital, SubShell, Shell
│   │
│   │  Composite systems
│   ├── atomic_system.py    AtomicSystem (base container)
│   ├── molecule.py         Molecule
│   ├── crystal.py          Crystal
│   │
│   │  Basis functions
│   ├── basis_functions.py  ContractedGaussianTypeOrbital, CGTOBasis
│   │
│   │  Integrals
│   ├── integrals/          Overlap, KineticEnergy, NuclearAttraction,
│   │                       TwoElectronRepulsion
│   │
│   │  Initialization
│   └── initialization/     QuantumCalculationContext, HartreeFock,
│                           RHF, UHF, ROHF, NuclearRepulsionEnergy
│
├── solvers/            Numerical algorithms and iterative methods.
│                       SCF loop machinery, DIIS convergence accelerator,
│                       eigensolvers, and concrete Hartree-Fock solvers
│                       (RestrictedHartreeFock, UnrestrictedHartreeFock,
│                       RestrictedOpenShellHartreeFock) and Kohn-Sham DFT
│                       solvers (RestrictedKohnSham, RestrictedOpenShellKohnSham,
│                       UnrestrictedKohnSham)
│                       with exchange-correlation functionals (SVWN, PBE,
│                       B3LYP).
│
└── utilities/          Shared mathematical utilities.
                        Pure-math helpers used across models and solvers:
                        Boys function, double factorial, normalisation
                        constants, Hermite expansion coefficients, and
                        Cartesian component generation.
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# -- Data structures --------------------------------------------------------
from .models.atom import Atom
from .models.electron import Orbital, Shell, SpinOrbital, SubShell

# -- Constants --------------------------------------------------------------
from .environment.constants.natural.atoms_data import (
    ANGSTROM_TO_BOHR,
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)

# -- I/O --------------------------------------------------------------------
from .environment.io.basis_set import BasisSet, Pople
from .environment.io.coordinates import CartesianCoordinates, Coordinates
from .environment.io.input_data import InputData

# -- Configuration ----------------------------------------------------------
from .environment.configuration import Configuration

# -- Composite systems ------------------------------------------------------
from .models.atomic_system import AtomicSystem
from .models.crystal import Crystal
from .models.molecule import Molecule

# -- Theory: initialization & basis functions --------------------------------
from .models.initialization import (
    RHF,
    RKS,
    ROHF,
    UHF,
    UKS,
    DensityFunctionalTheory,
    HartreeFock,
    QuantumCalculationContext,
)
from .models.basis_functions import ContractedGaussianTypeOrbital
from .models.integrals import (
    KineticEnergy,
    NuclearAttraction,
    NumericalGrid,
    Overlap,
    TwoElectronRepulsion,
)

__all__ = [
    "Atom",
    "AtomicSystem",
    "SpinOrbital",
    "Orbital",
    "SubShell",
    "Shell",
    "ATOMS_SYMBOLS_Z_TO_SYMBOL",
    "ATOMS_SYMBOLS_SYMBOL_TO_Z",
    "ANGSTROM_TO_BOHR",
    "BasisSet",
    "Pople",
    "InputData",
    "Coordinates",
    "CartesianCoordinates",
    "Molecule",
    "Crystal",
    "QuantumCalculationContext",
    "HartreeFock",
    "RHF",
    "UHF",
    "ROHF",
    "DensityFunctionalTheory",
    "RKS",
    "UKS",
    "ContractedGaussianTypeOrbital",
    "KineticEnergy",
    "NuclearAttraction",
    "NumericalGrid",
    "Overlap",
    "TwoElectronRepulsion",
]
