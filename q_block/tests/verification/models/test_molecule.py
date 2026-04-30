"""Unit tests for compute.models.molecule module.

Tests cover:
- GTO population via Molecule golden reference
- Molecule charge calculations
- Molecule electron count calculations
- Validation of multiplicity and electron parity
- AtomicSystem charge properties

All tests use pytest with parametrize, no test classes.
"""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import pytest

from q_block.compute import Atom
from q_block.compute.environment.constants.natural.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)
from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.models.atomic_system import AtomicSystem
from q_block.compute.models.electron import Shell, SpinOrbital
from q_block.compute.models.molecule import Molecule
from q_block.tests.verification.environment.constants import (
    BASIS_FILES,
    BASIS_ROOT,
    CHARGE_VALUES,
    DEFAULT_BASIS,
    GOLDEN_ROOT,
    charge_ids,
)

# ======================================================================
# Helper Functions
# ======================================================================


def _serialize_atom_for_test(atom: Atom, basis_name: str) -> Dict[str, Any]:
    """Serialize Atom into golden-reference comparable structure.

    :param atom: Atom to serialize.
    :type atom: Atom
    :param basis_name: Name of the basis set.
    :type basis_name: str
    :returns: Dictionary with atom data for comparison.
    :rtype: Dict[str, Any]
    """

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
    """Generate (basis_file, atom_symbol) pairs for pytest parametrization.

    :yields: Tuple of (basis_file, atom_symbol).
    :rtype: Iterator[Tuple[str, str]]
    """
    for basis_file in BASIS_FILES:
        basis_name = Path(basis_file).stem
        golden_path = GOLDEN_ROOT / f"{basis_name}.json"

        with golden_path.open() as f:
            golden = json.load(f)

        for symbol in golden["atoms"]:
            yield basis_file, symbol


def _make_input_data(atoms_spec: List[Tuple[str, int]]) -> InputData:
    """Build InputData from list of (symbol, charge) tuples.

    All atoms are placed at origin with DEFAULT_BASIS basis set.

    :param atoms_spec: List of (symbol, charge) pairs.
    :type atoms_spec: List[Tuple[str, int]]
    :returns: Populated InputData instance.
    :rtype: InputData
    """
    atom_data: List = []
    for symbol, charge in atoms_spec:
        atom_data.append([symbol, 0.0, 0.0, 0.0, DEFAULT_BASIS, charge])

    inp: InputData = InputData()
    inp.from_script(atom_data=atom_data, atom_prefix="T")
    return inp


def _total_n_electrons(atoms_spec: List[Tuple[str, int]]) -> int:
    """Calculate expected total electron count: sum(Z_i + q_i).

    :param atoms_spec: List of (symbol, charge) pairs.
    :type atoms_spec: List[Tuple[str, int]]
    :returns: Total electron count.
    :rtype: int
    """
    return sum(ATOMS_SYMBOLS_SYMBOL_TO_Z[sym] + q for sym, q in atoms_spec)


# ======================================================================
# GTO Golden Reference Tests
# ======================================================================


@pytest.mark.parametrize(
    "basis_file, symbol",
    list(_iter_golden_test_cases()),
    ids=lambda p: p if isinstance(p, str) else None,
)
def test_molecule_golden_gto_population(basis_file: str, symbol: str) -> None:
    """Verify GTO population via Molecule for single atom against golden reference.

    :param basis_file: Gaussian basis set filename.
    :type basis_file: str
    :param symbol: Atomic symbol (e.g., "H", "C", "Fe").
    :type symbol: str
    """
    basis_path: Path = BASIS_ROOT / basis_file
    basis_name: str = basis_path.stem

    golden_path: Path = GOLDEN_ROOT / f"{basis_name}.json"
    with golden_path.open() as f:
        golden = json.load(f)

    basis: Pople = Pople(filepath=str(basis_path))
    golden_atom: Dict = golden["atoms"][symbol]

    atom: Atom = Atom(
        atomic_number=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
        basis_set=basis,
    )

    row: Dict = {
        "atom_id": "1",
        "atomic_number": atom.atomic_number,
        "symbol": symbol,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "atom": atom,
    }
    input_data: InputData = InputData(atoms=pd.DataFrame([row]))

    n_el: int = atom.atomic_number
    mult: int = 1 if n_el % 2 == 0 else 2
    molecule: Molecule = Molecule(input_data=input_data, multiplicity=mult)
    assert len(molecule.atoms) == 1

    snapshot: Dict = _serialize_atom_for_test(atom=atom, basis_name=basis_name)

    assert (
        snapshot == golden_atom
    ), f"GTO population mismatch for atom {symbol} in basis {basis_name}"


# ======================================================================
# AtomicSystem Charge Tests
# ======================================================================


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_atomic_system_charge_single_atom(charge: int) -> None:
    """Verify AtomicSystem.charge equals single atom's charge.

    :param charge: Charge to assign to the atom.
    :type charge: int
    """
    inp: InputData = _make_input_data([("O", charge)])
    system: AtomicSystem = AtomicSystem(input_data=inp)
    assert system.charge == charge


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
    ids=[
        f"q1={q1},q2={q2}" for q1, q2 in itertools.product(CHARGE_VALUES, repeat=2)
    ],
)
def test_atomic_system_charge_two_atoms(q1: int, q2: int) -> None:
    """Verify AtomicSystem.charge equals sum of per-atom charges (H₂).

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    inp: InputData = _make_input_data([("H", q1), ("H", q2)])
    system: AtomicSystem = AtomicSystem(input_data=inp)
    assert system.charge == q1 + q2


# ======================================================================
# Molecule n_electrons - Single Atom Tests
# ======================================================================


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_molecule_n_electrons_single_oxygen(charge: int) -> None:
    """Verify Molecule.n_electrons for single O atom with various charges.

    O has Z=8, so n_electrons = 8 + charge. Multiplicity chosen to match parity.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    mult: int = 1 if n_el % 2 == 0 else 2
    inp: InputData = _make_input_data([("O", charge)])
    mol: Molecule = Molecule(input_data=inp, multiplicity=mult)
    assert mol.n_electrons == 8 + charge


# ======================================================================
# Molecule n_electrons - Two Atoms Tests
# ======================================================================


def _valid_h2_charge_combos() -> List[Tuple[int, int, int]]:
    """Return (q1, q2, mult) tuples where total n_electrons for H₂ >= 0.

    H has Z=1, so n_electrons = (1 + q1) + (1 + q2) = 2 + q1 + q2.
    """
    results: List[Tuple[int, int, int]] = []
    for q1, q2 in itertools.product(CHARGE_VALUES, repeat=2):
        n_el = 2 + q1 + q2
        if n_el < 0:
            continue
        mult = 1 if n_el % 2 == 0 else 2
        if n_el == 0:
            mult = 1
        results.append((q1, q2, mult))
    return results


@pytest.mark.parametrize(
    "q1, q2, mult",
    _valid_h2_charge_combos(),
    ids=[
        f"q1={q1},q2={q2},mult={mult}" for q1, q2, mult in _valid_h2_charge_combos()
    ],
)
def test_molecule_n_electrons_h2_charge_combos(q1: int, q2: int, mult: int) -> None:
    """Verify Molecule.n_electrons for H₂ with all valid charge combinations.

    Expected: (1 + q1) + (1 + q2) = 2 + q1 + q2.

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    :param mult: Spin multiplicity.
    :type mult: int
    """
    inp: InputData = _make_input_data([("H", q1), ("H", q2)])
    mol: Molecule = Molecule(input_data=inp, multiplicity=mult)
    expected: int = 2 + q1 + q2
    assert mol.n_electrons == expected
    assert mol.charge == q1 + q2


# ======================================================================
# Molecule n_electrons - Three Atoms (Water-like) Tests
# ======================================================================


def _valid_water_charge_combos() -> List[Tuple[int, int, int, int]]:
    """Return (qO, qH1, qH2, mult) where total n_electrons >= 0."""
    results: List[Tuple[int, int, int, int]] = []
    for qo, qh1, qh2 in itertools.product(CHARGE_VALUES, repeat=3):
        n_el = (8 + qo) + (1 + qh1) + (1 + qh2)
        if n_el < 0:
            continue
        mult = 1 if n_el % 2 == 0 else 2
        if n_el == 0:
            mult = 1
        results.append((qo, qh1, qh2, mult))
    return results


@pytest.mark.parametrize(
    "qo, qh1, qh2, mult",
    _valid_water_charge_combos(),
)
def test_molecule_n_electrons_water_charge_combos(
    qo: int, qh1: int, qh2: int, mult: int
) -> None:
    """Verify Molecule.n_electrons for water-like O-H-H with valid charge combos.

    :param qo: Charge on oxygen atom.
    :type qo: int
    :param qh1: Charge on first hydrogen atom.
    :type qh1: int
    :param qh2: Charge on second hydrogen atom.
    :type qh2: int
    :param mult: Spin multiplicity.
    :type mult: int
    """
    inp: InputData = _make_input_data([("O", qo), ("H", qh1), ("H", qh2)])
    mol: Molecule = Molecule(input_data=inp, multiplicity=mult)
    expected: int = (8 + qo) + (1 + qh1) + (1 + qh2)
    assert mol.n_electrons == expected
    assert mol.charge == qo + qh1 + qh2


# ======================================================================
# Molecule Validation - Negative Electrons Tests
# ======================================================================


def _negative_h2_charge_combos() -> List[Tuple[int, int]]:
    """Return (q1, q2) pairs where total n_electrons for H₂ is negative."""
    results: List[Tuple[int, int]] = []
    for q1, q2 in itertools.product(CHARGE_VALUES, repeat=2):
        n_el = 2 + q1 + q2
        if n_el < 0:
            results.append((q1, q2))
    return results


@pytest.mark.parametrize(
    "q1, q2",
    _negative_h2_charge_combos(),
    ids=[f"q1={q1},q2={q2}" for q1, q2 in _negative_h2_charge_combos()],
)
def test_molecule_negative_electrons_raises(q1: int, q2: int) -> None:
    """Verify Molecule construction raises ValueError when n_electrons < 0.

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    inp: InputData = _make_input_data([("H", q1), ("H", q2)])
    with pytest.raises(ValueError, match="Negative electron count"):
        Molecule(input_data=inp)


# ======================================================================
# Molecule Validation - Multiplicity Tests
# ======================================================================


def test_molecule_multiplicity_parity_mismatch_raises() -> None:
    """Verify Molecule raises when multiplicity incompatible with n_electrons.

    Water has 10 electrons (even). Multiplicity 2 requires odd n_electrons.
    """
    inp: InputData = _make_input_data([("O", 0), ("H", 0), ("H", 0)])
    with pytest.raises(ValueError, match="parity mismatch"):
        Molecule(input_data=inp, multiplicity=2)


def test_molecule_multiplicity_exceeds_electrons_raises() -> None:
    """Verify Molecule raises when multiplicity implies more unpaired than total.

    Single H (Z=1, charge=0) -> 1 electron. Multiplicity 4 needs 3 unpaired.
    """
    inp: InputData = _make_input_data([("H", 0)])
    with pytest.raises(ValueError, match="unpaired electrons"):
        Molecule(input_data=inp, multiplicity=4)


# ======================================================================
# Molecule total_atomic_number Tests
# ======================================================================


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_molecule_total_atomic_number_independent_of_charge(charge: int) -> None:
    """Verify total_atomic_number is sum of Z values, independent of charge.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    mult: int = 1 if n_el % 2 == 0 else 2
    inp: InputData = _make_input_data([("O", charge)])
    mol: Molecule = Molecule(input_data=inp, multiplicity=mult)
    assert mol.total_atomic_number == 8
