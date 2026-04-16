"""
Electronic density-based methods.

This module contains methods that work with the electron density ρ(r)
rather than the many-electron wavefunction. The primary approach is
Kohn-Sham DFT, which maps the interacting system onto a non-interacting
reference system sharing the same ground-state density.

Submodules:
    kohn_sham: Kohn-Sham DFT methods (RKS, UKS)
    functionals: Exchange-correlation functionals (LDA, GGA, hybrid)
"""
