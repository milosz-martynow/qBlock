"""
Methods module for electronic structure calculations.

Tree structure of implemented and planned methods:
==================================================

q_block/methods/
├── __init__.py
├── base.py                          # ElectronicStructureMethod ABC
│
├── algorithms/                      # Shared computational procedures
│   ├── __init__.py
│   ├── scf.py                       # SCF loop, convergence logic
│   ├── diis.py                      # DIIS convergence accelerator
│   ├── diagonalization.py           # Eigensolvers
│   └── davidson.py                  # Davidson algorithm (CI, EOM-CC)
│
├── wavefunction/                    # Ψ-based methods
│   ├── __init__.py
│   ├── base.py                      # WavefunctionMethod ABC
│   │
│   ├── mean_field/                  # Single determinant / SCF-based
│   │   ├── __init__.py
│   │   ├── base.py                  # MeanFieldMethod ABC
│   │   │
│   │   ├── hartree_fock/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # HartreeFock ABC
│   │   │   ├── rhf.py               # Restricted HF
│   │   │   ├── uhf.py               # Unrestricted HF
│   │   │   ├── rohf.py              # Restricted open-shell HF
│   │   │   └── fock_builder.py      # HF Fock matrix builder
│   │   │
│   │   └── mcscf/                   # Multi-configuration SCF
│   │       ├── __init__.py
│   │       ├── base.py              # MCSCF ABC
│   │       ├── casscf.py            # Complete active space SCF
│   │       ├── rasscf.py            # Restricted active space SCF
│   │       └── gasscf.py            # Generalized active space SCF
│   │
│   └── post_hf/                     # Post-Hartree-Fock (correlation)
│       ├── __init__.py
│       ├── base.py                  # PostHartreeFock ABC
│       │
│       ├── perturbation/            # Perturbation theory
│       │   ├── __init__.py
│       │   ├── base.py              # PerturbationTheory ABC
│       │   ├── mp2.py               # 2nd order Møller-Plesset
│       │   ├── mp3.py
│       │   ├── mp4.py
│       │   └── caspt2.py            # Multi-reference PT2
│       │
│       ├── ci/                      # Configuration Interaction
│       │   ├── __init__.py
│       │   ├── base.py              # ConfigurationInteraction ABC
│       │   ├── cis.py               # Singles (excited states)
│       │   ├── cid.py               # Doubles
│       │   ├── cisd.py              # Singles + Doubles
│       │   ├── cisdt.py             # + Triples
│       │   ├── mrci.py              # Multi-reference CI
│       │   └── full_ci.py           # Full CI (exact)
│       │
│       └── coupled_cluster/         # Coupled Cluster
│           ├── __init__.py
│           ├── base.py              # CoupledCluster ABC
│           ├── ccd.py               # Doubles
│           ├── ccsd.py              # Singles + Doubles
│           ├── ccsd_t.py            # + Perturbative Triples
│           ├── ccsdt.py             # Full Triples
│           └── eom_ccsd.py          # Equation of motion (excited)
│
└── electronic_density/              # ρ(r)-based methods
    ├── __init__.py
    ├── base.py                      # DensityBasedMethod ABC
    │
    └── kohn_sham/                   # Kohn-Sham DFT (orbital-based)
        ├── __init__.py
        ├── base.py                  # KohnShamDFT ABC
        ├── ks_solver.py             # KS-SCF implementation
        ├── fock_builder.py          # KS Fock matrix (H + J + Vxc)
        │
        └── functionals/             # Exchange-correlation functionals
            ├── __init__.py
            ├── base.py              # Functional ABC
            │
            ├── lda/                 # Local density approximation
            │   ├── __init__.py
            │   ├── base.py          # LDA ABC
            │   ├── svwn.py
            │   └── vwn5.py
            │
            ├── gga/                 # Generalized gradient approx.
            │   ├── __init__.py
            │   ├── base.py          # GGA ABC
            │   ├── pbe.py
            │   ├── blyp.py
            │   ├── bp86.py
            │   └── pw91.py
            │
            ├── meta_gga/            # Meta-GGA (+ kinetic density)
            │   ├── __init__.py
            │   ├── base.py          # MetaGGA ABC
            │   ├── tpss.py
            │   ├── m06l.py
            │   ├── scan.py
            │   └── r2scan.py
            │
            ├── hybrid/              # Hybrid (+ HF exchange)
            │   ├── __init__.py
            │   ├── base.py          # HybridFunctional ABC
            │   ├── b3lyp.py
            │   ├── pbe0.py
            │   ├── hse06.py         # Range-separated
            │   └── m06_2x.py
            │
            └── double_hybrid/       # Double hybrid (+ MP2 correlation)
                ├── __init__.py
                ├── base.py          # DoubleHybrid ABC
                ├── b2plyp.py
                └── pwpb95.py
   


Module organization summary:
============================

Path                                     | Contains           | Description
---------------------------------------- | ------------------ | ---------------------------------------------------
methods/                                 | Root               | Base class for all electronic structure methods
methods/algorithms/                      | Shared procedures  | Reusable algorithms (SCF, DIIS, eigensolvers, ...)
methods/wavefunction/                    | Ψ-based            | Methods solving for the wavefunction directly
methods/wavefunction/mean_field/         | SCF single-det     | Self-consistent field single-determinant methods
methods/wavefunction/post_hf/            | Correlation        | Methods that add electron correlation to HF
methods/electronic_density/              | ρ(r)-based         | Methods based on electron density functional
methods/electronic_density/kohn_sham/    | KS-DFT             | Kohn-Sham DFT with auxiliary orbital framework
"""
