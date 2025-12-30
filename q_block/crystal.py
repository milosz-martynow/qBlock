"""Crystalline atomic system container.

This module defines the :class:`Crystal` class, a minimal placeholder
for crystalline systems that extends :class:`q_block.atomic_system.AtomicSystem`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .atomic_system import AtomicSystem


class Crystal(AtomicSystem):
    """Container for atoms forming a crystalline system.

    This class is a minimal specialisation of
    :class:`q_block.atomic_system.AtomicSystem` intended to represent
    crystals or periodic atomic arrangements. At this stage it does not
    introduce any additional behaviour beyond its base class.

    :param atoms: Optional tabular description of atoms in the
        crystalline system.
    :type atoms: Optional[pd.DataFrame]
    """

    def __init__(self, atoms: Optional[pd.DataFrame] = None) -> None:
        super().__init__(atoms=atoms)
