"""Top-level test package for the qBlock project.

This package contains unit tests organized to mirror the q_block codebase structure:

- tests/io/: Tests for q_block.io module (coordinates, input_data)
- tests/solvers/: Tests for q_block.solvers module (DIIS, diagonalisation, error metrics, SCF)
- tests/models/: Tests for q_block.models module (atom, electron)
- tests/systems/: Tests for q_block.models module (molecule)
- tests/theory/: Tests for q_block.models module
  - tests/theory/initialization/: Tests for quantum calculation context classes
  - tests/theory/integrals/: Tests for integral calculations
- verification_data/: Expected results and golden reference data
"""

__all__ = []
