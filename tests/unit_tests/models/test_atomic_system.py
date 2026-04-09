"""Unit tests for q_block.models.atomic_system module.

Tests cover:
- AtomicSystem construction with and without InputData
- Charge computation from per-atom charges
- Atoms list access

All tests use pytest with parametrize, no test classes.
"""

from typing import List

import pytest

from q_block.io.input_data import InputData
from q_block.models.atomic_system import AtomicSystem


def test_atomic_system_default_construction() -> None:
    """Verify AtomicSystem can be constructed without arguments."""
    system = AtomicSystem()
    assert system.input_data is not None


def test_atomic_system_with_input_data() -> None:
    """Verify AtomicSystem stores InputData."""
    inp = InputData()
    system = AtomicSystem(input_data=inp)
    assert system.input_data is inp


def _make_system(
    atom_data: List[list],
    multiplicity: int = 1,
) -> AtomicSystem:
    """Build an AtomicSystem from a list of [symbol, x, y, z, ...] rows.

    :param atom_data: Rows accepted by :meth:`InputData.from_script`.
    :type atom_data: List[list]
    :param multiplicity: Spin multiplicity.
    :type multiplicity: int
    :returns: Fully populated AtomicSystem.
    :rtype: AtomicSystem
    """
    inp = InputData()
    inp.from_script(atom_data)
    return AtomicSystem(input_data=inp, multiplicity=multiplicity)


def test_charge_neutral_single_atom() -> None:
    """Neutral single atom should have charge 0."""
    system = _make_system([["H", 0.0, 0.0, 0.0]])
    assert system.charge == 0


def test_charge_neutral_two_atoms() -> None:
    """Two neutral atoms should have total charge 0."""
    system = _make_system([
        ["H", 0.0, 0.0, 0.0],
        ["He", 1.0, 0.0, 0.0],
    ])
    assert system.charge == 0


@pytest.mark.parametrize(
    "charge, expected",
    [(-1, -1), (0, 0), (1, 1), (2, 2)],
    ids=["anion", "neutral", "cation_+1", "cation_+2"],
)
def test_charge_single_atom(charge: int, expected: int) -> None:
    """Single atom with explicit charge.

    :param charge: Formal charge assigned to the atom.
    :type charge: int
    :param expected: Expected system charge.
    :type expected: int
    """
    system = _make_system([["H", 0.0, 0.0, 0.0, None, charge]])
    assert system.charge == expected


def test_charge_mixed_charges() -> None:
    """System with mixed per-atom charges sums correctly."""
    system = _make_system([
        ["Na", 0.0, 0.0, 0.0, None, 1],
        ["Cl", 2.0, 0.0, 0.0, None, -1],
    ])
    assert system.charge == 0


def test_charge_multiple_positive() -> None:
    """Multiple positively charged atoms."""
    system = _make_system([
        ["Li", 0.0, 0.0, 0.0, None, 1],
        ["Li", 2.0, 0.0, 0.0, None, 1],
    ])
    assert system.charge == 2
