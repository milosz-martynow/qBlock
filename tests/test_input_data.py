"""Unit tests for q_block.input_data.InputData helpers.

These tests verify that InputData.from_script and InputData.from_xyz_file
produce consistent pandas.DataFrame structures and Atom objects.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from q_block import AtomicSystem
from q_block.input_data import InputData


DATA_DIR = Path("tests/verification_data/geometries")


@pytest.mark.parametrize(
    "filename, expected_n",
    [
        ("H2.xyz", 2),
        ("water.xyz", 3),
    ],
)
def test_input_data_from_xyz_file_atom_count(
    filename: str, expected_n: int
) -> None:
    path = DATA_DIR / filename
    input_data = InputData()
    input_data.from_xyz_file(xyz_path=path, atom_prefix="T")

    assert isinstance(input_data, InputData)
    assert len(input_data) == expected_n


def test_input_data_from_script_structure_and_atoms() -> None:
    atoms_py = [
        ["O", 0.000000, 0.000000, 0.000000],
        ["H", 0.758602, 0.000000, 0.504284],
        ["H", -0.758602, 0.000000, 0.504284],
    ]

    input_data = InputData()
    input_data.from_script(atom_data=atoms_py, atom_prefix="S")
    df = input_data.atoms

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

    # data
    assert df["symbol"].tolist() == ["O", "H", "H"]
    assert df["atomic_number"].tolist() == [8, 1, 1]
    assert df["atom_id"].tolist() == ["S1", "S2", "S3"]

    atoms_col = df["atom"].tolist()
    assert all(atom is not None for atom in atoms_col)
    assert len({id(atom) for atom in atoms_col}) == len(atoms_col)

    system = AtomicSystem(input_data=input_data)
    assert len(system) == 3
    for _, row in system.input_data.atoms.iterrows():
        atom = row["atom"]
        x, y, z = atom.coordinates.as_tuple()  # type: ignore[union-attr]
        assert row["x"] == pytest.approx(x)
        assert row["y"] == pytest.approx(y)
        assert row["z"] == pytest.approx(z)


@pytest.mark.parametrize(
    "bad_symbol",
    ["x", "HE", "", "C1"],
)
def test_from_script_invalid_symbol_raises(bad_symbol: str) -> None:
    atoms_py = [[bad_symbol, 0.0, 0.0, 0.0]]
    input_data = InputData()
    with pytest.raises(ValueError):
        input_data.from_script(atom_data=atoms_py)


def test_from_xyz_file_invalid_atom_count(tmp_path: Path) -> None:
    xyz_content = """2
Comment
H 0.0 0.0 0.0
"""
    xyz_file = tmp_path / "bad.xyz"
    xyz_file.write_text(xyz_content, encoding="utf-8")

    input_data = InputData()
    with pytest.raises(ValueError):
        input_data.from_xyz_file(xyz_path=xyz_file)
