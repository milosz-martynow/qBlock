"""
Solvers module for electronic structure calculations.

Tree structure of implemented and planned methods:
==================================================

compute/solvers/
├── __init__.py
├── base.py                          # ElectronicStructureMethod base class
├── scf.py                           # SCF loop, convergence logic
├── diis.py                          # DIIS convergence accelerator
├── diagonalization.py               # Eigensolvers
├── davidson.py                      # Davidson algorithm (CI, EOM-CC)
│
├── wavefunction/                    # Ψ-based methods
│   ├── __init__.py
│   ├── base.py                      # WavefunctionMethod base class
│   │
│   ├── hartree_fock/                # Hartree-Fock methods
│   │   ├── __init__.py
│   │   ├── rhf.py                   # Restricted HF (closed-shell)
│   │   ├── uhf.py                   # Unrestricted HF (open-shell)
│   │   └── rohf.py                  # Restricted open-shell HF
│   │
│   ├── mcscf/                       # Multi-configuration SCF
│   │   ├── __init__.py
│   │   ├── casscf.py                # Complete active space SCF
│   │   ├── rasscf.py                # Restricted active space SCF
│   │   └── gasscf.py                # Generalized active space SCF
│   │
│   ├── perturbation/                # Perturbation theory
│   │   ├── __init__.py
│   │   ├── mp2.py                   # 2nd order Møller-Plesset
│   │   ├── mp3.py
│   │   ├── mp4.py
│   │   └── caspt2.py                # Multi-reference PT2
│   │
│   ├── ci/                          # Configuration Interaction
│   │   ├── __init__.py
│   │   ├── cis.py                   # Singles (excited states)
│   │   ├── cid.py                   # Doubles
│   │   ├── cisd.py                  # Singles + Doubles
│   │   ├── cisdt.py                 # + Triples
│   │   ├── mrci.py                  # Multi-reference CI
│   │   └── full_ci.py               # Full CI (exact)
│   │
│   └── coupled_cluster/             # Coupled Cluster
│       ├── __init__.py
│       ├── ccd.py                   # Doubles
│       ├── ccsd.py                  # Singles + Doubles
│       ├── ccsd_t.py                # + Perturbative Triples
│       ├── ccsdt.py                 # Full Triples
│       └── eom_ccsd.py              # Equation of motion (excited)
│
└── electronic_density/              # ρ(r)-based methods
    ├── __init__.py
    ├── base.py                      # DensityBasedMethod base class
    │
    ├── kohn_sham/                   # Kohn-Sham DFT (orbital-based)
    │   ├── __init__.py
    │   └── ks_solver.py             # KS-SCF implementation
    │
    ├── orbital_free/                # Orbital-free DFT
    │   ├── __init__.py
    │   └── thomas_fermi.py          # Thomas-Fermi model
    │
    └── functionals/                 # Exchange-correlation functionals
        ├── __init__.py
        ├── base.py                  # Functional base class
        │
        ├── lda/                     # Local density approximation
        │   ├── __init__.py
        │   ├── svwn.py
        │   └── vwn5.py
        │
        ├── gga/                     # Generalized gradient approx.
        │   ├── __init__.py
        │   ├── pbe.py
        │   ├── blyp.py
        │   ├── bp86.py
        │   └── pw91.py
        │
        ├── meta_gga/                # Meta-GGA (+ kinetic density)
        │   ├── __init__.py
        │   ├── tpss.py
        │   ├── m06l.py
        │   ├── scan.py
        │   └── r2scan.py
        │
        ├── hybrid/                  # Hybrid (+ HF exchange)
        │   ├── __init__.py
        │   ├── b3lyp.py
        │   ├── pbe0.py
        │   ├── hse06.py             # Range-separated
        │   └── m06_2x.py
        │
        └── double_hybrid/           # Double hybrid (+ MP2 correlation)
            ├── __init__.py
            ├── b2plyp.py
            └── pwpb95.py


Module organization summary:
============================

Path                                     | Contains             | Description
---------------------------------------- | -------------------- | ---------------------------------------------------
solvers/                                 | Root + algorithms    | Base class and shared algorithms (SCF, DIIS, etc.)
solvers/wavefunction/                    | Ψ-based methods      | Methods solving for the wavefunction directly
solvers/wavefunction/hartree_fock/       | HF methods           | Single-determinant mean-field approximation
solvers/wavefunction/mcscf/              | MCSCF methods        | Multi-configurational self-consistent field
solvers/wavefunction/perturbation/       | PT methods           | Møller-Plesset perturbation theory (MP2, MP3, ...)
solvers/wavefunction/ci/                 | CI methods           | Configuration interaction (CIS, CISD, Full CI, ...)
solvers/wavefunction/coupled_cluster/    | CC methods           | Coupled cluster (CCSD, CCSD(T), ...)
solvers/electronic_density/              | ρ(r)-based methods   | Methods based on electron density functional
solvers/electronic_density/kohn_sham/    | KS-DFT               | Kohn-Sham DFT with auxiliary orbital framework
solvers/electronic_density/orbital_free/ | OF-DFT               | Orbital-free DFT without Kohn-Sham orbitals
solvers/electronic_density/functionals/  | XC functionals       | Exchange-correlation functionals (LDA, GGA, ...)
"""

from .calculation_error_metric import CalculationErrorMetric
from .diagonalisation import diagonalise_fock, diagonalise_single
from .diis import DIIS
from .scf import SCF

__all__ = [
    "CalculationErrorMetric",
    "DIIS",
    "SCF",
    "diagonalise_fock",
    "diagonalise_single",
]
