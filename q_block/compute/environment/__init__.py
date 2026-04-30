"""Environment module for qBlock.

This module provides the foundational infrastructure for qBlock calculations:

- **constants**: Natural and numerical constants (atomic data, basis sets)
- **io**: Input/output interfaces (basis set readers, coordinate parsers,
  data containers)
- **configuration**: Input configuration reader (plain-text config files)
- **logs**: Centralised logging configuration
"""

from q_block.compute.environment.configuration import Configuration

__all__ = [
    "Configuration",
]
