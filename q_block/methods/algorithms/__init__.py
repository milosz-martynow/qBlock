"""
Shared computational algorithms for electronic structure methods.

This module contains reusable algorithms that are shared across different
electronic structure methods (both wavefunction and density-based).

Modules:
    scf: Self-consistent field loop and convergence logic
    diis: Direct Inversion in the Iterative Subspace convergence accelerator
    diagonalization: Eigensolvers for Fock/Kohn-Sham matrices
    davidson: Davidson algorithm for large eigenvalue problems (CI, EOM-CC)
    ...
"""
