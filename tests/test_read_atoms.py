"""Unit tests for q_block.read_atoms helpers.

These tests verify that populate_from_script and populate_from_xyz_file
produce consistent pandas.DataFrame structures and Atom objects.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from q_block import AtomicSystem, populate_from_script, populate_from_xyz_file


DATA_DIR = Path("tests/verification_data/geometries")


@pytest.mark.parametrize(
    "filename, expected_n",
    [
        ("H2.xyz", 2),
        ("water.xyz", 3),
    ],
)
def test_populate_from_xyz_file_atom_count(filename: str, expected_n: int) -> None:
    """XYZ reader should respect the declared atom count.

    This test only checks that the function:

    * reads the XYZ file without error for valid inputs,
    * returns a DataFrame with the same number of rows as the declared
      atom count.

    More detailed checks of DataFrame structure and Atom objects are
    performed in populate_from_script tests, since
    populate_from_xyz_file delegates final construction to
    :func:`populate_from_script`.

    :param filename: Name of the xyz file under the geometries directory.
    :type filename: str
    :param expected_n: Expected number of atoms.
    :type expected_n: int
    """
    path = DATA_DIR / filename
    df = populate_from_xyz_file(path, atom_prefix="T")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == expected_n


def test_populate_from_script_structure_and_atoms() -> None:
    """populate_from_script must create a fully structured DataFrame.

    This test checks column names, content of symbol and atomic_number,
    and that the "atom" column holds distinct Atom objects with
    coordinates consistent with the numeric columns.
    """
    atoms_py = [
        ["O", 0.000000, 0.000000, 0.000000],
        ["H", 0.758602, 0.000000, 0.504284],
        ["H", -0.758602, 0.000000, 0.504284],
    ]

    df = populate_from_script(atoms_py, atom_prefix="S")

    # shape and columns
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        "atom_id",
        "atomic_number",
        "symbol",
        "x",
        "y",
        "z",
        "atom",
    ]

    # content checks
    assert df["symbol"].tolist() == ["O", "H", "H"]
    assert df["atomic_number"].tolist() == [8, 1, 1]

    # atom_id prefix and ordering
    assert df["atom_id"].tolist() == ["S1", "S2", "S3"]

    # Atoms column is not empty and contains unique objects
    atoms_col = df["atom"].tolist()
    assert all(atom is not None for atom in atoms_col)
    assert len({id(atom) for atom in atoms_col}) == len(atoms_col)

    # Check that DataFrame coordinates match Atom coordinates via AtomicSystem
    system = AtomicSystem(df)
    assert len(system) == 3
    for (_, row) in system.atoms.iterrows():
        atom = row["atom"]
        x, y, z = atom.coordinates.as_tuple()  # type: ignore[union-attr]
        assert row["x"] == pytest.approx(x)
        assert row["y"] == pytest.approx(y)
        assert row["z"] == pytest.approx(z)


@pytest.mark.parametrize(
    "bad_symbol",
    ["x", "HE", "", "C1"],
)
def test_populate_from_script_invalid_symbol_raises(bad_symbol: str) -> None:
    """Invalid atomic symbols must raise ValueError.

    :param bad_symbol: Symbol string expected to be invalid.
    :type bad_symbol: str
    """
    atoms_py = [[bad_symbol, 0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        populate_from_script(atoms_py)


def test_populate_from_xyz_file_invalid_atom_count(tmp_path: Path) -> None:
    """Declared atom count not matching coordinate lines should raise.

    :param tmp_path: Temporary directory fixture provided by pytest.
    :type tmp_path: pathlib.Path
    """
    xyz_content = """2
Comment
H 0.0 0.0 0.0
"""
    xyz_file = tmp_path / "bad.xyz"
    xyz_file.write_text(xyz_content, encoding="utf-8")

    with pytest.raises(ValueError):
        populate_from_xyz_file(xyz_file)
