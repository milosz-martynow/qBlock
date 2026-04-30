# qBlock Requirements

## Software Requirements

1. qBlock shall be a Python software library for research on the electronic structure of atomic systems.

2. qBlock shall implement the SCF procedure to solve the nonlinear eigenvalue problem iteratively until convergence of the density matrix and total energy.

3. qBlock shall support Hartree-Fock theory in three variants: RHF, UHF, ROHF.

3. qBlock shall support Kohn-Sham Density Functional Theory in two variants: RKS, UKS.

4. qBlock shall support DFT with at least one LDA functional, one GGA functional, and one hybrid functional. NOTE: examples of implementations are: LDA functional in SVMN implementation, GGA functional in PBE implementation, hybrid functional in B3LYP implementation.

5. qBlock shall represent atomic systems electrons using Basis Sets. NOTE: examples of Basis Sets implementation are: CGTO and STO Pople basis sets read from standard Pople basis set files in GBS format e.g. STO-3G, 3-21G, 6-31G, 6-311++G**.

6. qBlock shall compute one-electron integrals: overlap, kinetic energy, and nuclear attraction.

7. qBlock shall compute two-electron repulsion integrals. NOTE: example of ERI implementation might be Obara-Saika or Boys-function-based scheme

8. qBlock shall allow to algorithmic acceleration of SCF convergence. NOTE: example of algorithmic acceleration of SCF is DIIS algorithm.  

9. qBlock shall expose a configuration interface that allows the user to control calculation parameters through a Configuration File. NOTE: example of configuration parameters are: basis set, charge, multiplicity, SCF convergence threshold, maximum iterations, DIIS settings, error metric, log level.

10. qBlock shall accept molecular geometry input either as an inline block inside the Configuration File or as an external XYZ file.

11. qBlock shall save calculation results in both JSON and plain-text formats.

12. qBlock shall provide a `compute/` directory that contains the core computational library organised into clearly separated software blocks.

13. qBlock shall organise its computational part into clearly separated software blocks. NOTE: examples of such blocks are: `models`, `utilities`, `solvers`, `environment`.

13. qBlock shall implement a `models` block that contains all physical domain models and data structures. NOTE: examples of such models and data structures are: atomic and molecular systems, basis functions, quantum mechanical integrals, numerical integration grids, and calculation contexts.

14. qBlock shall implement a `utilities` block that contains shared functions used across other blocks but carrying no domain-specific state. NOTE: examples of such shared functions are: Boys function, Hermite expansion, normalization, double factorial.

15. qBlock shall implement a `solvers` block that contains all numerical algorithms and convergence methods. NOTE: examples of such algorithms and methods are: RHF, UHF, ROHF, RKS, UKS solvers, exchange-correlation functionals, DIIS, and diagonalisation.

16. qBlock shall implement an `environment` block that provides all infrastructure services. NOTE: examples of such infrastructure services are: physical and numerical constants, basis set data files, I/O interfaces, and runtime configuration.

17. qBlock shall place all domain constants in a dedicated `environment/constants` layer that is read-only during a calculation. `constants/natural` shall store nature-based constants (physical constants, atomic data); `constants/numerical` shall store numerical parameters and basis set files.

18. qBlock shall provide a `learn/` directory that contains runnable example scripts demonstrating each major capability without requiring modifications to the library code, and documentation artefacts. NOTE: examples of major capabilities are: integral computations, SCF solvers, I/O pipeline. NOTE: examples of documentation artefacts are: architecture diagrams (UML component, block definition) and written reference material.

19. qBlock shall provide a `tests/` directory structured into verification and validation tests that confirm the correctness of the software at different levels of granularity.

20. qBlock shall include verification tests that confirm the correctness of individual classes and functions in isolation. NOTE: examples of verification test techniques are: fine-grained per-module unit tests using `pytest` and `@pytest.mark.parametrize`.

21. qBlock shall organise verification tests to mirror the directory and module structure of the `compute/` source code, so that each source module has a corresponding test module at the same relative path under `tests/verification/`.

22. qBlock shall include validation tests that verify computed results against known reference values within a defined numerical tolerance. NOTE: examples of validation tests are: end-to-end SCF calculations on atoms and small molecules verified against known ionization energies.

23. qBlock shall store golden reference data for verification and validation tests as versioned JSON files so that expected values are explicit and reproducible.

24. qBlock shall declare all runtime dependencies with pinned or bounded version constraints in `setup.py`.

25. qBlock shall enforce a consistent code style with project-level configuration files checked into the repository. NOTE: examples of static code style analysers are: Black, isort, Pylint.

26. qBlock shall name all classes using a noun in PascalCase. NOTE: examples of noun class names are: `NuclearAttraction`, `AtomicSystem`.

27. qBlock shall name all functions and methods using a verb in snake_case. NOTE: examples of verb function names are: `compute_energy`, `build_fock`.

28. qBlock shall name all constants using UPPER_SNAKE_CASE. NOTE: examples of constant names are: `HARTREE_TO_EV`, `ATOMS_SYMBOLS_Z_TO_SYMBOL`.

29. qBlock shall prefix all private class members and helper functions with a single underscore. NOTE: examples of private member names are: `_build_fock`, `_compute_matrix`.

30. qBlock shall document all public classes and functions with Sphinx/reST docstrings that include mathematical descriptions (LaTeX) of the underlying theory where applicable.

27. qBlock shall use abstract base classes and the Template Method design pattern in the SCF layer so that new solver variants can be added by subclassing without modifying the shared SCF loop. NOTE: examples of variations of solver are new DFT functionals or HF variants.

28. qBlock shall support molecules of arbitrary size and charge, as well as atomic systems, as inputs to all SCF solvers.

29. qBlock shall provide structured logging at configurable verbosity levels (DEBUG, INFO, WARNING, ERROR) so that the progress and results of a calculation can be monitored at runtime.


## Acronyms:
- HF - Hartree-Fock
- KS - Kohn-Sham
- DFT - Density Functional Theory
- SCF - Self-Consistent Field
- RHF - Restricted Hartree-Fock
- UHF - Unrestricted Hartree-Fock
- ROHF - Restricted Open Shell Hartree-Fock
- RKS - Restricted Kohn-Sham
- UKS - Unretricted Kohn-Sham
- LDA - Local Density Approximation
- GGA - Generalised Gradient Approximation
- SVWN - Slater-Vosko-Wilk-Nusair
- PBE - Perdew-Burke-Ernzerhof
- B3LYP - Becke three-parameter Lee-Yang-Parr
- CGTO - Contracted Gaussian-Type Orbital
- STO - Slater Type Orbital
- GBS - Gaussian Basis Set
- ERI - Two Electron Repulsion Integral
- DIIS - Direct Inversion in the Iterative Subspace 