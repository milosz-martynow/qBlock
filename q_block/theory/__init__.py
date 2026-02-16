"""Theory module for quantum-chemistry methods.

Subpackages
-----------
initialization
    Initialization classes for different calculation types (generic,
    Hartree-Fock, etc.).

basis_functions
    Contracted Gaussian-Type Orbital (CGTO) basis function
    representations for integral computation.
"""

from typing import Any

from .basis_functions import ContractedGaussianTypeOrbital

__all__ = [
    "Initialization",
    "HartreeFock",
    "RHF",
    "UHF",
    "ROHF",
    "ContractedGaussianTypeOrbital",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute access to avoid circular imports.

    Importing :mod:`q_block.theory` should be lightweight so that other
    packages (e.g. :mod:`q_block.models`) can import CGTO utilities
    without triggering initialization code that imports Molecule/Atom.
    """
    if name in {"Initialization", "HartreeFock", "RHF", "UHF", "ROHF"}:
        from .initialization import Initialization, HartreeFock, RHF, UHF, ROHF

        return {
            "Initialization": Initialization,
            "HartreeFock": HartreeFock,
            "RHF": RHF,
            "UHF": UHF,
            "ROHF": ROHF,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
