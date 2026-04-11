"""Top-level test package for the qBlock project.

This package contains unit tests organized to mirror the compute codebase structure:

- tests/io/: Tests for compute.io module (coordinates, input_data)
- tests/solvers/: Tests for compute.solvers module (DIIS, diagonalisation, error metrics, SCF)
- tests/models/: Tests for compute.models module (atom, electron)
- tests/systems/: Tests for compute.models module (molecule)
- tests/theory/: Tests for compute.models module
  - tests/theory/initialization/: Tests for quantum calculation context classes
  - tests/theory/integrals/: Tests for integral calculations
- verification_data/: Expected results and golden reference data
"""

__all__ = []
