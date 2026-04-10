"""Environment module for qBlock.

This module provides the foundational infrastructure for qBlock calculations:

- **constants**: Natural and numerical constants (atomic data, basis sets)
- **io**: Input/output interfaces (basis set readers, coordinate parsers, data containers)
- **configuration**: Runtime configuration management (paths, settings)

The environment module encapsulates all external data sources, system settings,
and interfaces that define the calculation environment.
"""

from q_block.environment.configuration import (
    CalculationDefaults,
    Configuration,
    OutputSettings,
    PathConfiguration,
)

__all__ = [
    "Configuration",
    "PathConfiguration",
    "CalculationDefaults",
    "OutputSettings",
]
