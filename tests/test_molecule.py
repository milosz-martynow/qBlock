"""Golden-reference tests for GTO population via Molecule.

This module validates that :class:`q_block.systems.molecule.Molecule` correctly
populates Gaussian-type orbital (GTO) data on :class:`q_block.models.atom.Atom`
instances using the same golden-reference data as the original
Atom-level implementation.

It also tests charge and electron-count calculations across all
combinations of per-atom charges (+2, +1, 0, -1, -2).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import pytest

from q_block import Atom
from q_block.constants.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)
from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.systems.atomic_system import AtomicSystem
from q_block.systems.molecule import Molecule

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")
GOLDEN_ROOT: Path = Path("./tests/verification_data/gto_population")

BASIS_FILES = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311++Gss.gbs",
]


def _serialize_atom_for_test(atom: Atom, basis_name: str) -> Dict[str, Any]:
    """Serialize Atom into golden-reference comparable structure.

    This helper mirrors the structure used in the original Atom tests
    and is intentionally duplicated here to keep the Molecule tests
    self-contained.
    """

    # lokalny import, aby uniknąć zbędnych zależności w czasie importu modułu testowego
    from q_block.models.electron import Shell, SpinOrbital  # local import to avoid cycles

    orbitals: Dict[str, Dict[str, list]] = {}

    shells: Dict[int, Shell] = atom.shells
    for shell in shells.values():
        for subshell in shell.subshells:

            occupied_spinorbitals = [
                so
                for orb in subshell.orbitals
                for so in (orb.spin_up, orb.spin_down)
                if so.occupied
            ]

            if not occupied_spinorbitals:
                continue

            key = f"{subshell.n}{'spdf'[subshell.l]}"
            so: SpinOrbital = occupied_spinorbitals[0]

            orbitals[key] = {
                "exponents": so.data["exponents"],
                "contractions": so.data["contractions"],
            }

    return {
        "symbol": ATOMS_SYMBOLS_Z_TO_SYMBOL[atom.atomic_number],
        "Z": atom.atomic_number,
        "basis": basis_name,
        "orbitals": orbitals,
    }


def _iter_golden_test_cases() -> Iterator[Tuple[str, str]]:
    """Generate (basis_file, atom_symbol) pairs for pytest parametrization."""

    for basis_file in BASIS_FILES:
        basis_name = Path(basis_file).stem
        golden_path = GOLDEN_ROOT / f"{basis_name}.json"

        with golden_path.open() as f:
            golden = json.load(f)

        for symbol in golden["atoms"]:
            yield basis_file, symbol


@pytest.mark.parametrize(
    "basis_file, symbol",
    list(_iter_golden_test_cases()),
    ids=lambda p: p if isinstance(p, str) else None,
)
def test_molecule_golden_gto_population(basis_file: str, symbol: str) -> None:
    """Verify GTO population via Molecule for a single atom.

    For each (basis_file, symbol) pair, this test constructs a single
    :class:`Atom` with the corresponding basis-set fragment, wraps it in
    :class:`InputData`, and builds a :class:`Molecule`. The Molecule
    initialisation triggers GTO population on the atom, and the
    resulting structure is compared against the golden-reference data.

    :param basis_file: Gaussian basis set filename.
    :type basis_file: str

    :param symbol: Atomic symbol (e.g. ``"H"``, ``"C"``, ``"Fe"``).
    :type symbol: str
    """

    basis_path = BASIS_ROOT / basis_file
    basis_name = basis_path.stem

    golden_path = GOLDEN_ROOT / f"{basis_name}.json"
    with golden_path.open() as f:
        golden = json.load(f)

    basis = Pople(filepath=str(basis_path))
    golden_atom = golden["atoms"][symbol]

    atom = Atom(
        atomic_number=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
        basis_set=basis,
    )

    row = {
        "atom_id": "1",
        "atomic_number": atom.atomic_number,
        "symbol": symbol,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "atom": atom,
    }
    input_data = InputData(atoms=pd.DataFrame([row]))

    # Choose multiplicity compatible with the atom's electron count
    n_el = atom.atomic_number
    mult = 1 if n_el % 2 == 0 else 2
    molecule = Molecule(input_data=input_data, multiplicity=mult)
    assert len(molecule.atoms) == 1

    snapshot = _serialize_atom_for_test(atom=atom, basis_name=basis_name)

    assert snapshot == golden_atom, (
        f"GTO population mismatch for atom {symbol} " f"in basis {basis_name}"
    )


# ==============================================================================
# Helpers for charge / n_electrons tests
# ==============================================================================

CHARGE_VALUES: List[int] = [+2, +1, 0, -1, -2]

# Default basis set for charge/electron tests (lightweight Pople set)
_DEFAULT_BASIS = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")


def _make_input_data(
    atoms_spec: List[Tuple[str, int]],
) -> InputData:
    """Build an InputData from a list of (symbol, charge) tuples.

    All atoms are placed at the origin with 6-31G basis set.

    :param atoms_spec: List of ``(symbol, charge)`` pairs.
    :returns: Populated :class:`InputData`.
    """
    atom_data = []
    for symbol, charge in atoms_spec:
        atom_data.append([symbol, 0.0, 0.0, 0.0, _DEFAULT_BASIS, charge])

    inp = InputData()
    inp.from_script(atom_data=atom_data, atom_prefix="T")
    return inp


def _total_n_electrons(atoms_spec: List[Tuple[str, int]]) -> int:
    """Expected total electron count: sum(Z_i + q_i)."""
    return sum(
        ATOMS_SYMBOLS_SYMBOL_TO_Z[sym] + q for sym, q in atoms_spec
    )


# ==============================================================================
# AtomicSystem.charge tests
# ==============================================================================


@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_atomic_system_charge_single_atom(charge: int) -> None:
    """AtomicSystem.charge equals the single atom's charge."""
    inp = _make_input_data([("O", charge)])
    system = AtomicSystem(input_data=inp)
    assert system.charge == charge


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
)
def test_atomic_system_charge_two_atoms(q1: int, q2: int) -> None:
    """AtomicSystem.charge equals the sum of per-atom charges (H₂)."""
    inp = _make_input_data([("H", q1), ("H", q2)])
    system = AtomicSystem(input_data=inp)
    assert system.charge == q1 + q2


# ==============================================================================
# Molecule.n_electrons – single atom, all charges
# ==============================================================================


@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_molecule_n_electrons_single_oxygen(charge: int) -> None:
    """Molecule.n_electrons for a single O atom with various charges.

    O has Z=8, so n_electrons = 8 + charge.  All values from 6 to 10
    are non-negative, so no validation error should occur.
    Multiplicity chosen to match parity.
    """
    n_el = 8 + charge
    mult = 1 if n_el % 2 == 0 else 2
    inp = _make_input_data([("O", charge)])
    mol = Molecule(input_data=inp, multiplicity=mult)
    assert mol.n_electrons == 8 + charge


# ==============================================================================
# Molecule.n_electrons – two atoms, all 25 charge combinations
# ==============================================================================


def _valid_h2_charge_combos() -> List[Tuple[int, int, int]]:
    """Return (q1, q2, mult) tuples where total n_electrons for H₂ is >= 0.

    H has Z=1, so n_electrons = (1 + q1) + (1 + q2) = 2 + q1 + q2.
    Require 2 + q1 + q2 >= 0 and parity-compatible multiplicity.
    """
    results = []
    for q1, q2 in itertools.product(CHARGE_VALUES, repeat=2):
        n_el = 2 + q1 + q2
        if n_el < 0:
            continue
        mult = 1 if n_el % 2 == 0 else 2
        if n_el == 0:
            mult = 1
        results.append((q1, q2, mult))
    return results


@pytest.mark.parametrize("q1, q2, mult", _valid_h2_charge_combos())
def test_molecule_n_electrons_h2_charge_combos(
    q1: int, q2: int, mult: int,
) -> None:
    """Molecule.n_electrons for H₂ with all valid charge combinations.

    Expected: (1 + q1) + (1 + q2) = 2 + q1 + q2.
    """
    inp = _make_input_data([("H", q1), ("H", q2)])
    mol = Molecule(input_data=inp, multiplicity=mult)
    expected = 2 + q1 + q2
    assert mol.n_electrons == expected
    assert mol.charge == q1 + q2


# ==============================================================================
# Molecule.n_electrons – three atoms (water-like), all 125 combos
# ==============================================================================


def _valid_water_charge_combos() -> List[Tuple[int, int, int, int]]:
    """Return (qO, qH1, qH2, mult) where total n_electrons >= 0 and parity-compatible."""
    results = []
    for qo, qh1, qh2 in itertools.product(CHARGE_VALUES, repeat=3):
        n_el = (8 + qo) + (1 + qh1) + (1 + qh2)
        if n_el < 0:
            continue
        mult = 1 if n_el % 2 == 0 else 2
        if n_el == 0:
            mult = 1
        results.append((qo, qh1, qh2, mult))
    return results


@pytest.mark.parametrize("qo, qh1, qh2, mult", _valid_water_charge_combos())
def test_molecule_n_electrons_water_charge_combos(
    qo: int, qh1: int, qh2: int, mult: int,
) -> None:
    """Molecule.n_electrons for water-like O-H-H with all valid charge combos."""
    inp = _make_input_data([("O", qo), ("H", qh1), ("H", qh2)])
    mol = Molecule(input_data=inp, multiplicity=mult)
    expected = (8 + qo) + (1 + qh1) + (1 + qh2)
    assert mol.n_electrons == expected
    assert mol.charge == qo + qh1 + qh2


# ==============================================================================
# Molecule validation: negative electron count raises
# ==============================================================================


def _negative_h2_charge_combos() -> List[Tuple[int, int]]:
    """Return (q1, q2) pairs where total n_electrons for H₂ is negative."""
    results = []
    for q1, q2 in itertools.product(CHARGE_VALUES, repeat=2):
        n_el = 2 + q1 + q2
        if n_el < 0:
            results.append((q1, q2))
    return results


@pytest.mark.parametrize("q1, q2", _negative_h2_charge_combos())
def test_molecule_negative_electrons_raises(q1: int, q2: int) -> None:
    """Molecule construction should raise ValueError when n_electrons < 0."""
    inp = _make_input_data([("H", q1), ("H", q2)])
    with pytest.raises(ValueError, match="Negative electron count"):
        Molecule(input_data=inp)


# ==============================================================================
# Molecule validation: multiplicity parity mismatch raises
# ==============================================================================


def test_molecule_multiplicity_parity_mismatch_raises() -> None:
    """Molecule should raise when multiplicity is incompatible with n_electrons."""
    # Water has 10 electrons (even). Multiplicity 2 requires odd n_electrons.
    inp = _make_input_data([("O", 0), ("H", 0), ("H", 0)])
    with pytest.raises(ValueError, match="parity mismatch"):
        Molecule(input_data=inp, multiplicity=2)


def test_molecule_multiplicity_exceeds_electrons_raises() -> None:
    """Molecule should raise when multiplicity implies more unpaired than total."""
    # Single H (Z=1, charge=0) -> 1 electron. Multiplicity 4 needs 3 unpaired.
    inp = _make_input_data([("H", 0)])
    with pytest.raises(ValueError, match="unpaired electrons"):
        Molecule(input_data=inp, multiplicity=4)


# ==============================================================================
# Molecule.total_atomic_number
# ==============================================================================


@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_molecule_total_atomic_number_independent_of_charge(
    charge: int,
) -> None:
    """total_atomic_number is sum of Z values and does not depend on charge."""
    n_el = 8 + charge
    mult = 1 if n_el % 2 == 0 else 2
    inp = _make_input_data([("O", charge)])
    mol = Molecule(input_data=inp, multiplicity=mult)
    assert mol.total_atomic_number == 8
