"""Molecular atomic system container.

This module defines the :class:`Molecule` class, a minimal placeholder
for molecular systems that extends :class:`q_block.atomic_system.AtomicSystem`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .atomic_system import AtomicSystem


class Molecule(AtomicSystem):
    """Container for atoms forming a single molecule.

    This class is a minimal specialisation of
    :class:`q_block.atomic_system.AtomicSystem` intended to represent
    molecular systems. At this stage it does not introduce any
    additional behaviour beyond its base class.

    :param atoms: Optional tabular description of atoms in the
        molecular system.
    :type atoms: Optional[pd.DataFrame]
    """

    def __init__(self, atoms: Optional[pd.DataFrame] = None) -> None:
        super().__init__(atoms=atoms)
