"""Unit tests for compute.io.input_data module.

Tests cover:
- InputData.from_xyz_file: reading XYZ geometry files
- InputData.from_script: creating input data from Python lists
- DataFrame structure and atom object creation
- Validation of atomic symbols

All tests use pytest with parametrize, no test classes.
"""

from pathlib import Path
from typing import List

import pandas as pd
import pytest

from compute import AtomicSystem
from compute.environment.io.input_data import InputData
from tests.unit_tests.environment.constants import GEOMETRIES_DIR

# ======================================================================
# InputData.from_xyz_file Tests
# ======================================================================


@pytest.mark.parametrize(
    "filename, expected_atom_count",
    [
        ("H2.xyz", 2),
        ("water.xyz", 3),
    ],
    ids=["H2", "water"],
)
def test_input_data_from_xyz_file_atom_count(
    filename: str, expected_atom_count: int
) -> None:
    """Verify from_xyz_file correctly reads atom count from XYZ files.

    :param filename: Name of XYZ file in the verification_data/geometries folder.
    :type filename: str
    :param expected_atom_count: Expected number of atoms in the file.
    :type expected_atom_count: int
    """
    path: Path = GEOMETRIES_DIR / filename
    input_data: InputData = InputData()
    input_data.from_xyz_file(xyz_path=path, atom_prefix="T")

    assert isinstance(input_data, InputData)
    assert len(input_data) == expected_atom_count


def test_from_xyz_file_invalid_atom_count(tmp_path: Path) -> None:
    """Verify from_xyz_file raises ValueError when declared count mismatches.

    Creates a temporary XYZ file that declares 2 atoms but only contains 1,
    which should raise ValueError.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    xyz_content: str = """2
Comment
H 0.0 0.0 0.0
"""
    xyz_file: Path = tmp_path / "bad.xyz"
    xyz_file.write_text(xyz_content, encoding="utf-8")

    input_data: InputData = InputData()
    with pytest.raises(ValueError):
        input_data.from_xyz_file(xyz_path=xyz_file)


# ======================================================================
# InputData.from_script Tests
# ======================================================================


def test_input_data_from_script_structure_and_atoms() -> None:
    """Verify from_script creates correct DataFrame structure and Atom objects.

    Creates InputData from a Python list representing water molecule
    and validates DataFrame columns, content, and Atom object creation.
    """
    atoms_py: List[List] = [
        ["O", 0.000000, 0.000000, 0.000000],
        ["H", 0.758602, 0.000000, 0.504284],
        ["H", -0.758602, 0.000000, 0.504284],
    ]

    input_data: InputData = InputData()
    input_data.from_script(atom_data=atoms_py, atom_prefix="S")
    df: pd.DataFrame = input_data.atoms

    # Validate DataFrame shape and columns
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

    # Validate data content
    assert df["symbol"].tolist() == ["O", "H", "H"]
    assert df["atomic_number"].tolist() == [8, 1, 1]
    assert df["atom_id"].tolist() == ["S1", "S2", "S3"]

    # Validate Atom objects
    atoms_col: List = df["atom"].tolist()
    assert all(atom is not None for atom in atoms_col)
    assert len({id(atom) for atom in atoms_col}) == len(atoms_col)


def test_input_data_from_script_coordinates_match_atoms() -> None:
    """Verify coordinates in DataFrame match Atom.coordinates attributes.

    Creates an AtomicSystem and validates that row coordinates
    match the corresponding Atom object coordinates.
    """
    atoms_py: List[List] = [
        ["O", 0.000000, 0.000000, 0.000000],
        ["H", 0.758602, 0.000000, 0.504284],
        ["H", -0.758602, 0.000000, 0.504284],
    ]

    input_data: InputData = InputData()
    input_data.from_script(atom_data=atoms_py, atom_prefix="S")

    system: AtomicSystem = AtomicSystem(input_data=input_data)
    assert len(system) == 3

    for _, row in system.input_data.atoms.iterrows():
        atom = row["atom"]
        x, y, z = atom.coordinates.as_tuple()
        assert row["x"] == pytest.approx(x)
        assert row["y"] == pytest.approx(y)
        assert row["z"] == pytest.approx(z)


# ======================================================================
# from_script Validation Tests
# ======================================================================


@pytest.mark.parametrize(
    "bad_symbol, reason",
    [
        ("x", "lowercase single letter not valid"),
        ("HE", "uppercase multi-letter (should be He)"),
        ("", "empty string"),
        ("C1", "symbol with number suffix"),
    ],
    ids=["lowercase_x", "uppercase_HE", "empty", "with_number"],
)
def test_from_script_invalid_symbol_raises(bad_symbol: str, reason: str) -> None:
    """Verify from_script raises ValueError for invalid atomic symbols.

    :param bad_symbol: Invalid atomic symbol to test.
    :type bad_symbol: str
    :param reason: Description of why the symbol is invalid.
    :type reason: str
    """
    atoms_py: List[List] = [[bad_symbol, 0.0, 0.0, 0.0]]
    input_data: InputData = InputData()

    with pytest.raises(ValueError):
        input_data.from_script(atom_data=atoms_py)
