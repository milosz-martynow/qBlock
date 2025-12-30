"""Atomic system container for Atom objects.

This module defines the :class:`AtomicSystem` class, a lightweight,
quantum‑agnostic container for atoms with Cartesian coordinates.

The class stores atoms via an :class:`q_block.input_data.InputData`
instance, which itself wraps the canonical :class:`pandas.DataFrame` with
atom metadata and attached :class:`q_block.atom.Atom` instances.
"""

from __future__ import annotations

from typing import Optional

from q_block.input_data import InputData


class AtomicSystem:
    """Container for atoms in a generic quantum system.

    This class stores atom metadata and attached :class:`q_block.atom.Atom`
    instances using an :class:`q_block.input_data.InputData` object.

    The underlying :class:`pandas.DataFrame` used by :class:`InputData`
    must use the columns::

        ["atom_id", "atomic_number", "symbol", "x", "y", "z", "atom"]

    :param input_data: Optional prebuilt :class:`InputData` instance. When
                       provided, the object will wrap it directly. If
                       omitted, an empty :class:`InputData` with the
                       required header is created.
    :type input_data: Optional[InputData]
    """

    def __init__(self, input_data: Optional[InputData] = None) -> None:
        if input_data is None:
            input_data = InputData()
        self.input_data: InputData = input_data

    def __len__(self) -> int:
        """Return the number of atoms stored in the system."""
        return int(self.input_data.atoms.shape[0])

    def __repr__(self) -> str:
        return f"AtomicSystem(n_atoms={len(self)})"
