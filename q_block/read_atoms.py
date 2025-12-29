"""Helpers to read atoms from different sources into a pandas.DataFrame.

This module provides small, reusable functions that convert input data
(either a Python structure or an XYZ file) into a canonical
:pandas:`DataFrame` with the header::

    ["atom_id", "atomic_number", "symbol", "x", "y", "z", "atom"]

The functions are intentionally independent from :class:`AtomicSystem` so
that the same parsing code can be reused in tests or other modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import pandas as pd

from q_block.atom import Atom
from q_block.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z
from q_block.coordinates import CartesianCoordinates


def _check_atom_symbol(symbol_raw: Any) -> str:
    """Validate an atomic symbol string and return the canonical symbol.

    The function verifies that the provided symbol is a non-empty string,
    starts with an uppercase letter followed by optional lowercase
    alphabetic characters, and that the symbol exists in the project's
    atom lookup table ``ATOMS_SYMBOLS_SYMBOL_TO_Z``.

    :param symbol_raw: Object representing an atomic symbol (any type).
    :type symbol_raw: Any
    :returns: Canonicalized atomic symbol string (trimmed).
    :rtype: str
    :raises ValueError: If the symbol is empty, has incorrect casing/format,
                        or is not a known chemical symbol.
    """
    symbol = str(symbol_raw).strip()
    if not symbol:
        raise ValueError(f"Atomic symbol must be a non-empty string; got {symbol!r}")

    if not symbol[0].isupper() or (len(symbol) > 1 and not symbol[1:].islower()) or not symbol.isalpha():
        raise ValueError(
            f"Atomic symbol must start with a capital letter followed by lowercase letters: got {symbol!r}"
        )

    if symbol not in ATOMS_SYMBOLS_SYMBOL_TO_Z:
        raise ValueError(f"Unknown atomic symbol: {symbol!r}")

    return symbol


def populate_from_script(atom_data: List[List[Any]], atom_prefix: str = "") -> pd.DataFrame:
    """Create atoms DataFrame from a Python structure.

    The expected input format is::

        [[symbol, x, y, z], ...]

    The returned :class:`pandas.DataFrame` has columns::

        ["atom_id", "atomic_number", "symbol", "x", "y", "z", "atom"]

    :param atom_data: List of atomic records ``[symbol, x, y, z]``.
    :type atom_data: List[List[str, float, float, float]]
    :param atom_prefix: Optional prefix for generated atom identifiers.
    :type atom_prefix: str
    :returns: Populated atoms DataFrame.
    :rtype: pandas.DataFrame
    :raises ValueError: If the input data are malformed or contain unknown
                        chemical symbols.
    """
    rows: List[dict[str, Any]] = []

    for idx, entry in enumerate(atom_data):
        if not isinstance(entry, (list, tuple)) or len(entry) < 4:
            raise ValueError(
                f"Each entry must be [symbol, x, y, z]; got {entry!r} at index {idx}"
            )

        symbol = _check_atom_symbol(entry[0])

        try:
            coordinates = CartesianCoordinates.from_sequence(entry[1:4])
        except ValueError as exc:
            raise ValueError(f"Coordinates must be convertible to float; got {entry[1:4]!r}") from exc

        x, y, z = coordinates.as_tuple()

        atomic_number = ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol]
        atom_id = f"{atom_prefix}{idx + 1}"

        atom = Atom(atomic_number=atomic_number, coordinates=coordinates)

        rows.append(
            {
                "atom_id": atom_id,
                "atomic_number": atomic_number,
                "symbol": symbol,
                "x": x,
                "y": y,
                "z": z,
                "atom": atom,
            }
        )

    df = pd.DataFrame(rows, columns=["atom_id", "atomic_number", "symbol", "x", "y", "z", "atom"])
    return df


def populate_from_xyz_file(xyz_path: Union[str, Path], atom_prefix: str = "") -> pd.DataFrame:
    """Create atoms DataFrame from an XYZ file.

    The function accepts the conventional XYZ format::

        N
        comment line (may be empty)
        Symbol x y z
        ... (N lines)

    Empty comment lines are allowed. The function returns the same
    DataFrame format as :func:`populate_from_script`.

    :param xyz_path: Path to the XYZ file.
    :type xyz_path: Union[str, Path]
    :param atom_prefix: Optional prefix for generated atom identifiers.
    :type atom_prefix: str
    :returns: Populated atoms DataFrame.
    :rtype: pandas.DataFrame
    :raises ValueError: If the file is malformed or the declared atom count
                        does not match the number of coordinate lines found.
    """
    path = Path(xyz_path)
    text = path.read_text(encoding="utf-8")

    raw_lines = [ln.rstrip("\n\r") for ln in text.splitlines()]

    if len(raw_lines) < 3:
        raise ValueError("XYZ file must contain at least three lines (N, comment, and one coordinate line)")

    first_line = raw_lines[0].strip()
    try:
        n_atoms = int(first_line)
    except ValueError:
        raise ValueError("First line of XYZ file must be an integer atom count")

    # Keep comment line (raw_lines[1]) even if empty; collect coordinate lines ignoring blank lines.
    coord_lines = [ln.strip() for ln in raw_lines[2:] if ln.strip()]
    if len(coord_lines) != n_atoms:
        raise ValueError(f"XYZ file declares {n_atoms} atoms but only {len(coord_lines)} coordinate lines were found")

    data: List[List[Any]] = []
    for i in range(n_atoms):
        parts = coord_lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ coordinate line {i + 3}: {coord_lines[i]!r}")

        symbol = _check_atom_symbol(parts[0])
        try:
            coordinates = CartesianCoordinates.from_sequence(parts[1:4])
            x, y, z = coordinates.as_tuple()
        except ValueError:
            raise ValueError(f"XYZ coordinates must be floats on line {i + 3}: {coord_lines[i]!r}")

        data.append([symbol, x, y, z])

    return populate_from_script(data, atom_prefix=atom_prefix)

