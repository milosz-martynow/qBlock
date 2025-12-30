"""Atomic system container for Atom objects.

This module defines the :class:`AtomicSystem` class, a lightweight,
quantum‑agnostic container for atoms with Cartesian coordinates. Low‑level
reading/parsing functions live in :mod:`q_block.read_input_files` and return the
canonical :class:`pandas.DataFrame` accepted by this class.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class AtomicSystem:
    """Container for atoms in a generic quantum system.

    This class stores atom metadata and attached :class:`q_block.atom.Atom`
    instances using a :class:`pandas.DataFrame` produced by the
    :mod:`q_block.read_input_files` helpers.

    The DataFrame must use the columns::

        ["atom_id", "atomic_number", "symbol", "x", "y", "z", "atom"]

    :param atoms: Optional prebuilt atoms DataFrame. When provided the
                  object will wrap the passed DataFrame directly without
                  copying. If omitted an empty DataFrame with the required
                  header is created.
    :type atoms: Optional[pandas.DataFrame]
    """

    def __init__(self, atoms: Optional[pd.DataFrame] = None) -> None:
        if atoms is None:
            atoms = pd.DataFrame(
                columns=[
                    "atom_id",
                    "atomic_number",
                    "symbol",
                    "x",
                    "y",
                    "z",
                    "atom",
                ]
            )
        self.atoms: pd.DataFrame = atoms

    def __len__(self) -> int:
        """Return the number of atoms stored in the system."""
        return int(self.atoms.shape[0])

    def __repr__(self) -> str:
        return f"AtomicSystem(n_atoms={len(self)})"
