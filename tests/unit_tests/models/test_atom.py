"""Unit tests for q_block.models.atom module.

Tests cover:
- Atom spin-orbital occupancy (pure Aufbau filling)
- GTO population via Molecule integration
- Open/closed shell classification
- Charge and electron count calculations
- Atom attributes and constants validation

All tests use pytest with parametrize, no test classes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import pytest

from q_block import Atom, Molecule
from q_block.environment.constants.natural.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    CLOSED_SHELL_ATOMS,
    OPEN_SHELL_ATOMS,
)
from q_block.environment.io.basis_set import Pople
from q_block.environment.io.input_data import InputData
from q_block.models.electron import Shell, SpinOrbital
from tests.unit_tests.environment.constants import (
    BASIS_FILES,
    BASIS_ROOT,
    CHARGE_VALUES,
    GOLDEN_ROOT,
    charge_ids,
)
from tests.unit_tests.verification_data.expected_atom_pure import EXPECTED_ATOM_PURE


# Type aliases for spin-orbital mappings
SpinKey = Tuple[int, int, int, float]
SpinMap = Dict[SpinKey, bool]


# ======================================================================
# Helper Functions
# ======================================================================


def extract_spin_map(atom: Atom) -> SpinMap:
    """Extract complete spin-orbital occupancy map from an Atom instance.

    :param atom: Fully initialized and filled Atom object.
    :type atom: Atom
    :returns: Mapping of (n, l, m, s) to occupied flag.
    :rtype: SpinMap
    """
    result: SpinMap = {}

    shells: Dict[int, Shell] = atom.shells
    for shell in shells.values():
        for subs in shell.subshells:
            for orb in subs.orbitals:
                up: SpinOrbital = orb.spin_up
                down: SpinOrbital = orb.spin_down

                result[(orb.n, orb.l, orb.m, up.s)] = up.occupied
                result[(orb.n, orb.l, orb.m, down.s)] = down.occupied

    return result


def count_spin_map(spin_map: SpinMap) -> int:
    """Count number of occupied spin orbitals in mapping.

    :param spin_map: Spin-orbital occupancy mapping.
    :type spin_map: SpinMap
    :returns: Number of occupied spin orbitals.
    :rtype: int
    """
    return sum(1 for occupied in spin_map.values() if occupied)


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

    for shell in atom.shells.values():
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
            so = occupied_spinorbitals[0]

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


# ======================================================================
# Pure Aufbau Spin-Orbital Tests
# ======================================================================


@pytest.mark.parametrize("atomic_number", range(1, 119))
def test_atom_pure_matches_expected(atomic_number: int) -> None:
    """Verify Atom with pure Aufbau filling matches expected spin map.

    :param atomic_number: Atomic number to test (1-118).
    :type atomic_number: int
    """
    expected_pure: SpinMap = EXPECTED_ATOM_PURE[atomic_number]

    atom: Atom = Atom(
        atomic_number=atomic_number,
        maximal_principal_quantum_number=7,
        empirical_exceptions=None,
    )
    atom.fill_occupancy()

    actual: SpinMap = extract_spin_map(atom)

    assert set(actual.keys()) == set(expected_pure.keys()), (
        f"Spin-orbital key mismatch for atomic_number={atomic_number}"
    )

    for key in actual:
        assert actual[key] == expected_pure[key], (
            f"Value mismatch at atomic_number={atomic_number} on spin orbital {key}"
        )

    assert count_spin_map(actual) == atomic_number, (
        f"Electron count mismatch for atomic_number={atomic_number}"
    )


# ======================================================================
# GTO Population Golden Reference Tests
# ======================================================================


@pytest.mark.parametrize(
    "basis_file, symbol",
    list(_iter_golden_test_cases()),
    ids=lambda p: p if isinstance(p, str) else None,
)
def test_golden_gto_population(basis_file: str, symbol: str) -> None:
    """Verify GTO population for single atom against golden reference.

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
        "x": atom.coordinates.x if atom.coordinates else 0.0,
        "y": atom.coordinates.y if atom.coordinates else 0.0,
        "z": atom.coordinates.z if atom.coordinates else 0.0,
        "atom": atom,
    }
    input_data: InputData = InputData(atoms=pd.DataFrame([row]))
    n_el: int = atom.atomic_number
    mult: int = 1 if n_el % 2 == 0 else 2
    Molecule(input_data=input_data, multiplicity=mult)

    snapshot: Dict = _serialize_atom_for_test(atom=atom, basis_name=basis_name)

    assert snapshot == golden_atom, (
        f"GTO population mismatch for atom {symbol} in basis {basis_name}"
    )


# ======================================================================
# Open/Closed Shell Tests
# ======================================================================


@pytest.mark.parametrize("atomic_number", CLOSED_SHELL_ATOMS)
def test_atom_closed_shell(atomic_number: int) -> None:
    """Verify closed-shell atoms have open_shell=False.

    Closed-shell atoms have all electrons paired (no unpaired electrons).
    Includes noble gases and atoms with completely filled subshells.

    :param atomic_number: Atomic number of atom to test.
    :type atomic_number: int
    """
    atom: Atom = Atom(atomic_number=atomic_number)
    symbol: str = ATOMS_SYMBOLS_Z_TO_SYMBOL[atomic_number]

    assert atom.open_shell is False, (
        f"Atom {symbol} (Z={atomic_number}) should be closed-shell "
        f"but open_shell={atom.open_shell}"
    )


@pytest.mark.parametrize("atomic_number", OPEN_SHELL_ATOMS)
def test_atom_open_shell(atomic_number: int) -> None:
    """Verify open-shell atoms have open_shell=True.

    Open-shell atoms have at least one unpaired electron.
    Includes radicals and atoms with partially filled subshells.

    :param atomic_number: Atomic number of atom to test.
    :type atomic_number: int
    """
    atom: Atom = Atom(atomic_number=atomic_number)
    symbol: str = ATOMS_SYMBOLS_Z_TO_SYMBOL[atomic_number]

    assert atom.open_shell is True, (
        f"Atom {symbol} (Z={atomic_number}) should be open-shell "
        f"but open_shell={atom.open_shell}"
    )


# Pure Aufbau closed-shell atoms (all subshells completely filled)
# Differs from empirical CLOSED_SHELL_ATOMS due to exceptions like Pd, Cu, Cr, etc.
PURE_AUFBAU_CLOSED_SHELL: List[int] = [
    2, 4, 10, 12, 18, 20, 30, 36, 38, 48, 54, 56, 70, 80, 86, 88, 102, 112, 118
]


def _generate_open_shell_test_cases() -> List[Tuple[int, bool]]:
    """Generate (atomic_number, expected_open_shell) for all 118 elements.

    Uses pure Aufbau filling rules (no empirical exceptions).
    Closed-shell atoms are those with completely filled subshells.

    :returns: List of tuples with atomic number and expected open_shell value.
    :rtype: List[Tuple[int, bool]]
    """
    return [
        (z, z not in PURE_AUFBAU_CLOSED_SHELL)
        for z in range(1, 119)
    ]


@pytest.mark.parametrize(
    "atomic_number, expected_open_shell",
    _generate_open_shell_test_cases(),
    ids=[f"Z{z}_{ATOMS_SYMBOLS_Z_TO_SYMBOL[z]}" for z in range(1, 119)],
)
def test_atom_open_shell_pure_aufbau(
    atomic_number: int, expected_open_shell: bool
) -> None:
    """Verify open_shell attribute with pure Aufbau filling for all 118 elements.

    :param atomic_number: Atomic number to test.
    :type atomic_number: int
    :param expected_open_shell: Expected value of open_shell attribute.
    :type expected_open_shell: bool
    """
    atom: Atom = Atom(atomic_number=atomic_number, empirical_exceptions=None)
    assert atom.open_shell is expected_open_shell


def test_atom_open_shell_after_refill() -> None:
    """Verify open_shell is correctly updated after calling fill_occupancy.

    The open_shell attribute should be recalculated each time fill_occupancy
    is called.
    """
    atom: Atom = Atom(atomic_number=6)  # Carbon, open shell
    assert atom.open_shell is True

    atom.fill_occupancy()
    assert atom.open_shell is True


# ======================================================================
# Open/Closed Shell Lists Validation Tests
# ======================================================================


def test_open_shell_lists_cover_all_elements() -> None:
    """Verify CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS cover all 118 elements.

    Every element from Z=1 (H) to Z=118 (Og) must appear in exactly one list.
    """
    all_elements: set = set(range(1, 119))
    closed_set: set = set(CLOSED_SHELL_ATOMS)
    open_set: set = set(OPEN_SHELL_ATOMS)

    combined: set = closed_set | open_set

    assert combined == all_elements, (
        f"Missing elements: {all_elements - combined}, "
        f"Extra elements: {combined - all_elements}"
    )


def test_open_shell_lists_no_overlap() -> None:
    """Verify CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS have no overlap.

    An atom cannot be both closed-shell and open-shell.
    """
    closed_set: set = set(CLOSED_SHELL_ATOMS)
    open_set: set = set(OPEN_SHELL_ATOMS)

    overlap: set = closed_set & open_set

    assert len(overlap) == 0, f"Atoms appear in both lists: {overlap}"


def test_open_shell_lists_counts() -> None:
    """Verify counts of closed and open shell atoms sum to 118."""
    n_closed: int = len(CLOSED_SHELL_ATOMS)
    n_open: int = len(OPEN_SHELL_ATOMS)

    assert n_closed + n_open == 118, (
        f"Expected 118 total elements, got {n_closed} closed + {n_open} open"
    )


def test_open_shell_lists_no_duplicates() -> None:
    """Verify neither list contains duplicate entries."""
    assert len(CLOSED_SHELL_ATOMS) == len(set(CLOSED_SHELL_ATOMS)), (
        "CLOSED_SHELL_ATOMS contains duplicates"
    )
    assert len(OPEN_SHELL_ATOMS) == len(set(OPEN_SHELL_ATOMS)), (
        "OPEN_SHELL_ATOMS contains duplicates"
    )


@pytest.mark.parametrize(
    "shell_list",
    [CLOSED_SHELL_ATOMS, OPEN_SHELL_ATOMS],
    ids=["closed_shell", "open_shell"],
)
def test_open_shell_lists_valid_atomic_numbers(shell_list: List[int]) -> None:
    """Verify all atomic numbers in lists are valid (1-118).

    :param shell_list: List of atomic numbers to validate.
    :type shell_list: List[int]
    """
    for z in shell_list:
        assert 1 <= z <= 118, f"Invalid atomic number: {z}"


def test_all_atoms_open_shell_consistency() -> None:
    """Verify every atom's open_shell attribute matches its list membership.

    Iterates over all 118 elements and verifies consistency.
    """
    closed_set: set = set(CLOSED_SHELL_ATOMS)
    open_set: set = set(OPEN_SHELL_ATOMS)

    for z in range(1, 119):
        atom: Atom = Atom(atomic_number=z)
        symbol: str = ATOMS_SYMBOLS_Z_TO_SYMBOL[z]

        if z in closed_set:
            assert atom.open_shell is False, (
                f"Atom {symbol} (Z={z}) is in CLOSED_SHELL_ATOMS but has open_shell=True"
            )
        elif z in open_set:
            assert atom.open_shell is True, (
                f"Atom {symbol} (Z={z}) is in OPEN_SHELL_ATOMS but has open_shell=False"
            )
        else:
            raise AssertionError(
                f"Atom {symbol} (Z={z}) is not in either list"
            )


# ======================================================================
# Element Group Classification Tests
# ======================================================================


@pytest.mark.parametrize(
    "atomic_number",
    [2, 10, 18, 36, 54, 86, 118],
    ids=["Z2_He", "Z10_Ne", "Z18_Ar", "Z36_Kr", "Z54_Xe", "Z86_Rn", "Z118_Og"],
)
def test_noble_gases_are_closed_shell(atomic_number: int) -> None:
    """Verify noble gases are correctly classified as closed-shell.

    Noble gases have completely filled outer shells and no unpaired electrons.

    :param atomic_number: Atomic number of noble gas.
    :type atomic_number: int
    """
    assert atomic_number in CLOSED_SHELL_ATOMS
    atom: Atom = Atom(atomic_number=atomic_number)
    assert atom.open_shell is False


@pytest.mark.parametrize(
    "atomic_number",
    [3, 11, 19, 37, 55, 87],
    ids=["Z3_Li", "Z11_Na", "Z19_K", "Z37_Rb", "Z55_Cs", "Z87_Fr"],
)
def test_alkali_metals_are_open_shell(atomic_number: int) -> None:
    """Verify alkali metals are correctly classified as open-shell.

    Alkali metals have one unpaired electron in their outer s orbital.

    :param atomic_number: Atomic number of alkali metal.
    :type atomic_number: int
    """
    assert atomic_number in OPEN_SHELL_ATOMS
    atom: Atom = Atom(atomic_number=atomic_number)
    assert atom.open_shell is True


@pytest.mark.parametrize(
    "atomic_number",
    [9, 17, 35, 53, 85, 117],
    ids=["Z9_F", "Z17_Cl", "Z35_Br", "Z53_I", "Z85_At", "Z117_Ts"],
)
def test_halogens_are_open_shell(atomic_number: int) -> None:
    """Verify halogens are correctly classified as open-shell.

    Halogens have one unpaired electron (p⁵ configuration).

    :param atomic_number: Atomic number of halogen.
    :type atomic_number: int
    """
    assert atomic_number in OPEN_SHELL_ATOMS
    atom: Atom = Atom(atomic_number=atomic_number)
    assert atom.open_shell is True


@pytest.mark.parametrize(
    "atomic_number",
    [4, 12, 20, 38, 56, 88],
    ids=["Z4_Be", "Z12_Mg", "Z20_Ca", "Z38_Sr", "Z56_Ba", "Z88_Ra"],
)
def test_alkaline_earth_metals_are_closed_shell(atomic_number: int) -> None:
    """Verify alkaline earth metals are correctly classified as closed-shell.

    Alkaline earth metals have filled s² outer orbital with all electrons paired.

    :param atomic_number: Atomic number of alkaline earth metal.
    :type atomic_number: int
    """
    assert atomic_number in CLOSED_SHELL_ATOMS
    atom: Atom = Atom(atomic_number=atomic_number)
    assert atom.open_shell is False


# ======================================================================
# Charge and Electron Count Tests
# ======================================================================


@pytest.mark.parametrize("charge", CHARGE_VALUES, ids=charge_ids())
def test_atom_charge_stored(charge: int) -> None:
    """Verify charge attribute is stored correctly on Atom.

    :param charge: Formal charge on the atom.
    :type charge: int
    """
    atom: Atom = Atom(atomic_number=8, charge=charge)
    assert atom.charge == charge


@pytest.mark.parametrize("charge", CHARGE_VALUES, ids=charge_ids())
def test_atom_n_electrons_with_charge(charge: int) -> None:
    """Verify n_electrons = Z + q for single atom across charge values.

    :param charge: Formal charge on the atom.
    :type charge: int
    """
    z: int = 8  # Oxygen
    atom: Atom = Atom(atomic_number=z, charge=charge)
    assert atom.n_electrons == z + charge


@pytest.mark.parametrize(
    "atomic_number",
    [1, 6, 8, 26, 79],
    ids=["Z1_H", "Z6_C", "Z8_O", "Z26_Fe", "Z79_Au"],
)
@pytest.mark.parametrize("charge", CHARGE_VALUES, ids=charge_ids())
def test_atom_n_electrons_various_elements(atomic_number: int, charge: int) -> None:
    """Verify n_electrons = Z + q for various elements and charges.

    :param atomic_number: Atomic number of the element.
    :type atomic_number: int
    :param charge: Formal charge on the atom.
    :type charge: int
    """
    atom: Atom = Atom(atomic_number=atomic_number, charge=charge)
    assert atom.n_electrons == atomic_number + charge


def test_atom_default_charge_is_zero() -> None:
    """Verify default charge is 0 (neutral atom)."""
    atom: Atom = Atom(atomic_number=6)
    assert atom.charge == 0
    assert atom.n_electrons == 6


@pytest.mark.parametrize("z", range(1, 119))
def test_atom_n_electrons_neutral_equals_atomic_number(z: int) -> None:
    """Verify neutral atom has n_electrons equal to atomic number.

    :param z: Atomic number to test.
    :type z: int
    """
    atom: Atom = Atom(atomic_number=z)
    assert atom.n_electrons == z


def test_atom_cation_fewer_electrons() -> None:
    """Verify cation has fewer electrons than neutral atom.

    Na+ (sodium cation): Z=11, charge=-1 -> N=10
    """
    atom: Atom = Atom(atomic_number=11, charge=-1)
    assert atom.n_electrons == 10


def test_atom_anion_more_electrons() -> None:
    """Verify anion has more electrons than neutral atom.

    Cl- (chloride): Z=17, charge=+1 -> N=18
    """
    atom: Atom = Atom(atomic_number=17, charge=+1)
    assert atom.n_electrons == 18
