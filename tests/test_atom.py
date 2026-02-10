import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
import pandas as pd

from q_block import Atom, Molecule
from q_block.constants.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    CLOSED_SHELL_ATOMS,
    OPEN_SHELL_ATOMS,
)
from q_block.io.basis_set import Pople
from q_block.models.electron import Shell, SpinOrbital
from q_block.io.input_data import InputData
from tests.verification_data.expected_atom_pure import EXPECTED_ATOM_PURE

SpinKey = Tuple[int, int, int, float]
SpinMap = Dict[SpinKey, bool]

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")
GOLDEN_ROOT: Path = Path("./tests/verification_data/gto_population")

BASIS_FILES = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311++Gss.gbs",
]


def extract_spin_map(atom: Atom) -> SpinMap:
    """
    Extract the complete spin-orbital occupancy map from an Atom instance.

    :param atom: a fully initialized & filled Atom object
    :returns: mapping of spin orbitals to occupied flags
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
    """
    Count number of electrons in a spin-orbital map.

    :param spin_map: spin-orbital occupancy
    :returns: number of occupied spin orbitals
    """
    return sum(1 for occupied in spin_map.values() if occupied)


def maps_equal(a: SpinMap, b: SpinMap) -> bool:
    """
    Compare two spin-orbital maps for exact match of keys and values.
    """
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if a[k] != b[k]:
            return False
    return True


@pytest.mark.parametrize("atomic_number", range(1, 119))
def test_atom_pure_matches_expected(
    atomic_number: int,
) -> None:
    """
    Test that Atom(atomic_number, pure mode) matches the expected static spin map.

    :param atomic_number: atomic number under test
    """
    expected_pure: SpinMap = EXPECTED_ATOM_PURE[atomic_number]

    atom = Atom(
        atomic_number=atomic_number,
        maximal_principal_quantum_number=7,
        empirical_exceptions=None,
    )
    atom.fill_occupancy()

    actual = extract_spin_map(atom)

    assert set(actual.keys()) == set(
        expected_pure.keys()
    ), f"Spin-orbital key mismatch for atomic_number={atomic_number}"

    for key in actual:
        assert (
            actual[key] == expected_pure[key]
        ), f"Value mismatch at atomic_number={atomic_number} on spin orbital {key}"

    assert (
        count_spin_map(actual) == atomic_number
    ), f"Electron count mismatch for atomic_number={atomic_number}"


@pytest.mark.parametrize("atomic_number", range(1, 119))
def test_atom_pure_matches_expected_for_exceptions(
    atomic_number: int,
) -> None:
    """
    Test that Atom(atomic_number, pure mode) matches the expected pure spin-orbital map,
    but **only for atoms that have empirical exceptions**.

    Meaning:
      - atomic_number is tested only if it appears in EXPECTED_ATOM_EMPIRICAL
      - For other atomic_number, this test is skipped

    :param atomic_number: atomic number under test
    """
    # Only test atoms that actually have empirical exceptions
    # if atomic_number not in EXPECTED_ATOM_EMPIRICAL:
    #     pytest.skip(f"atomic_number={atomic_number} has no empirical exception — skipping pure test.")

    expected_pure: SpinMap = EXPECTED_ATOM_PURE[atomic_number]

    atom = Atom(
        atomic_number=atomic_number,
        maximal_principal_quantum_number=7,
        empirical_exceptions=None,
    )
    atom.fill_occupancy()

    # Ensure the atom's state is correctly updated without relying on a return value
    actual = extract_spin_map(atom)

    # Keyset must match exactly
    assert set(actual.keys()) == set(
        expected_pure.keys()
    ), f"[PURE] Spin-orbital key mismatch for exception atom atomic_number={atomic_number}"

    # Values must match exactly
    for key in actual:
        assert (
            actual[key] == expected_pure[key]
        ), f"[PURE] Value mismatch at atomic_number={atomic_number} on spin orbital {key}"

    # Pure electron count must always equal atomic_number
    assert (
        count_spin_map(actual) == atomic_number
    ), f"[PURE] Electron count mismatch for atomic_number={atomic_number}"


"""
Golden-reference tests for GTO population.

===============================================================================
PURPOSE
===============================================================================

This test module verifies the correctness of GTO population on atoms
when driven by a molecular container (see :mod:`test_molecule`).

The historical implementation lived on :class:`Atom` as
``populate_spinorbitals_with_gto``; the logic has since moved to
:class:`q_block.molecule.Molecule`. The golden-reference serialization
helpers remain here for reuse.

Each golden-reference file corresponds to ONE basis set and contains
verified GTO assignments for ALL atoms defined in that basis.

Each (basis set, atom) pair is tested as an independent unit test.
===============================================================================
"""


def _serialize_atom_for_test(
    atom: Atom,
    basis_name: str,
) -> Dict[str, Any]:
    """Serialize Atom into golden-reference comparable structure."""

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
    """
    Generate (basis_file, atom_symbol) pairs for pytest parametrization.

    Yields
    ------
    (basis_file, atom_symbol)
        One test case per atom per basis set.
    """

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
def test_golden_gto_population(
    basis_file: str,
    symbol: str,
) -> None:
    """
    Verify GTO population for a single atom in a single basis set.

    This test compares the populated GTO data against manually verified
    golden-reference data.

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
    # Wrap the single atom into InputData/Molecule so that GTO population
    # is performed at the molecular level.
    row = {
        "atom_id": "1",
        "atomic_number": atom.atomic_number,
        "symbol": symbol,
        "x": atom.coordinates.x if atom.coordinates else 0.0,
        "y": atom.coordinates.y if atom.coordinates else 0.0,
        "z": atom.coordinates.z if atom.coordinates else 0.0,
        "atom": atom,
    }
    input_data = InputData(atoms=pd.DataFrame([row]))
    n_el = atom.atomic_number
    mult = 1 if n_el % 2 == 0 else 2
    Molecule(input_data=input_data, multiplicity=mult)

    snapshot = _serialize_atom_for_test(
        atom=atom,
        basis_name=basis_name,
    )

    assert snapshot == golden_atom, (
        f"GTO population mismatch for atom {symbol} " f"in basis {basis_name}"
    )


# ==============================================================================
# Tests for open_shell attribute
# ==============================================================================
# 
# CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS are imported from
# q_block.constants.atoms_data and contain all atoms from the periodic table.
# ==============================================================================


@pytest.mark.parametrize("atomic_number", CLOSED_SHELL_ATOMS)
def test_atom_closed_shell(atomic_number: int) -> None:
    """Test that closed-shell atoms have open_shell=False.
    
    Closed-shell atoms have all electrons paired (no unpaired electrons).
    This includes noble gases and atoms with completely filled subshells.
    
    :param atomic_number: Atomic number of atom to test.
    """
    atom = Atom(atomic_number=atomic_number)
    
    symbol = ATOMS_SYMBOLS_Z_TO_SYMBOL[atomic_number]
    assert atom.open_shell is False, (
        f"Atom {symbol} (Z={atomic_number}) should be closed-shell "
        f"but open_shell={atom.open_shell}"
    )


@pytest.mark.parametrize("atomic_number", OPEN_SHELL_ATOMS)
def test_atom_open_shell(atomic_number: int) -> None:
    """Test that open-shell atoms have open_shell=True.
    
    Open-shell atoms have at least one unpaired electron.
    This includes radicals and atoms with partially filled subshells.
    
    :param atomic_number: Atomic number of atom to test.
    """
    atom = Atom(atomic_number=atomic_number)
    
    symbol = ATOMS_SYMBOLS_Z_TO_SYMBOL[atomic_number]
    assert atom.open_shell is True, (
        f"Atom {symbol} (Z={atomic_number}) should be open-shell "
        f"but open_shell={atom.open_shell}"
    )


def test_atom_open_shell_pure_aufbau() -> None:
    """Test open_shell attribute with pure Aufbau filling (no empirical exceptions).
    
    When empirical_exceptions=None, the Aufbau principle is strictly followed.
    This should still correctly identify open/closed shell atoms.
    """
    # Helium - closed shell
    he = Atom(atomic_number=2, empirical_exceptions=None)
    assert he.open_shell is False
    
    # Hydrogen - open shell  
    h = Atom(atomic_number=1, empirical_exceptions=None)
    assert h.open_shell is True
    
    # Carbon - open shell (2p² has 2 unpaired)
    c = Atom(atomic_number=6, empirical_exceptions=None)
    assert c.open_shell is True
    
    # Nitrogen - open shell (2p³ has 3 unpaired)
    n = Atom(atomic_number=7, empirical_exceptions=None)
    assert n.open_shell is True
    
    # Neon - closed shell (full 2p⁶)
    ne = Atom(atomic_number=10, empirical_exceptions=None)
    assert ne.open_shell is False


def test_atom_open_shell_after_refill() -> None:
    """Test that open_shell is correctly updated after calling fill_occupancy again.
    
    The open_shell attribute should be recalculated each time fill_occupancy
    is called.
    """
    atom = Atom(atomic_number=6)  # Carbon, open shell
    assert atom.open_shell is True
    
    # Call fill_occupancy again - should still be open shell
    atom.fill_occupancy()
    assert atom.open_shell is True


# ==============================================================================
# Generic tests for CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS lists
# ==============================================================================


def test_open_shell_lists_cover_all_elements() -> None:
    """Test that CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS together cover all 118 elements.
    
    Every element from Z=1 (H) to Z=118 (Og) must appear in exactly one list.
    """
    all_elements = set(range(1, 119))
    closed_set = set(CLOSED_SHELL_ATOMS)
    open_set = set(OPEN_SHELL_ATOMS)
    
    combined = closed_set | open_set
    
    assert combined == all_elements, (
        f"Missing elements: {all_elements - combined}, "
        f"Extra elements: {combined - all_elements}"
    )


def test_open_shell_lists_no_overlap() -> None:
    """Test that CLOSED_SHELL_ATOMS and OPEN_SHELL_ATOMS have no overlap.
    
    An atom cannot be both closed-shell and open-shell.
    """
    closed_set = set(CLOSED_SHELL_ATOMS)
    open_set = set(OPEN_SHELL_ATOMS)
    
    overlap = closed_set & open_set
    
    assert len(overlap) == 0, (
        f"Atoms appear in both lists: {overlap}"
    )


def test_open_shell_lists_counts() -> None:
    """Test that the counts of closed and open shell atoms are correct.
    
    Total should be 118 elements.
    """
    n_closed = len(CLOSED_SHELL_ATOMS)
    n_open = len(OPEN_SHELL_ATOMS)
    
    assert n_closed + n_open == 118, (
        f"Expected 118 total elements, got {n_closed} closed + {n_open} open = {n_closed + n_open}"
    )


def test_open_shell_lists_no_duplicates() -> None:
    """Test that neither list contains duplicate entries."""
    assert len(CLOSED_SHELL_ATOMS) == len(set(CLOSED_SHELL_ATOMS)), (
        "CLOSED_SHELL_ATOMS contains duplicates"
    )
    assert len(OPEN_SHELL_ATOMS) == len(set(OPEN_SHELL_ATOMS)), (
        "OPEN_SHELL_ATOMS contains duplicates"
    )


def test_open_shell_lists_valid_atomic_numbers() -> None:
    """Test that all atomic numbers in both lists are valid (1-118)."""
    for z in CLOSED_SHELL_ATOMS:
        assert 1 <= z <= 118, f"Invalid atomic number in CLOSED_SHELL_ATOMS: {z}"
    
    for z in OPEN_SHELL_ATOMS:
        assert 1 <= z <= 118, f"Invalid atomic number in OPEN_SHELL_ATOMS: {z}"


def test_all_atoms_open_shell_consistency() -> None:
    """Test that every atom's open_shell attribute matches its list membership.
    
    Iterates over all 118 elements and verifies:
    - Atoms in CLOSED_SHELL_ATOMS have open_shell=False
    - Atoms in OPEN_SHELL_ATOMS have open_shell=True
    """
    closed_set = set(CLOSED_SHELL_ATOMS)
    open_set = set(OPEN_SHELL_ATOMS)
    
    for z in range(1, 119):
        atom = Atom(atomic_number=z)
        symbol = ATOMS_SYMBOLS_Z_TO_SYMBOL[z]
        
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
                f"Atom {symbol} (Z={z}) is not in either CLOSED_SHELL_ATOMS or OPEN_SHELL_ATOMS"
            )


def test_noble_gases_are_closed_shell() -> None:
    """Test that all noble gases are correctly classified as closed-shell.
    
    Noble gases have completely filled outer shells and no unpaired electrons.
    """
    noble_gases = [2, 10, 18, 36, 54, 86, 118]  # He, Ne, Ar, Kr, Xe, Rn, Og
    
    for z in noble_gases:
        assert z in CLOSED_SHELL_ATOMS, (
            f"Noble gas Z={z} should be in CLOSED_SHELL_ATOMS"
        )
        atom = Atom(atomic_number=z)
        assert atom.open_shell is False, (
            f"Noble gas Z={z} should have open_shell=False"
        )


def test_alkali_metals_are_open_shell() -> None:
    """Test that all alkali metals are correctly classified as open-shell.
    
    Alkali metals have one unpaired electron in their outer s orbital.
    """
    alkali_metals = [3, 11, 19, 37, 55, 87]  # Li, Na, K, Rb, Cs, Fr
    
    for z in alkali_metals:
        assert z in OPEN_SHELL_ATOMS, (
            f"Alkali metal Z={z} should be in OPEN_SHELL_ATOMS"
        )
        atom = Atom(atomic_number=z)
        assert atom.open_shell is True, (
            f"Alkali metal Z={z} should have open_shell=True"
        )


def test_halogens_are_open_shell() -> None:
    """Test that all halogens are correctly classified as open-shell.
    
    Halogens have one unpaired electron (p⁵ configuration).
    """
    halogens = [9, 17, 35, 53, 85, 117]  # F, Cl, Br, I, At, Ts
    
    for z in halogens:
        assert z in OPEN_SHELL_ATOMS, (
            f"Halogen Z={z} should be in OPEN_SHELL_ATOMS"
        )
        atom = Atom(atomic_number=z)
        assert atom.open_shell is True, (
            f"Halogen Z={z} should have open_shell=True"
        )


def test_alkaline_earth_metals_are_closed_shell() -> None:
    """Test that all alkaline earth metals are correctly classified as closed-shell.
    
    Alkaline earth metals have a filled s² outer orbital with all electrons paired.
    """
    alkaline_earth = [4, 12, 20, 38, 56, 88]  # Be, Mg, Ca, Sr, Ba, Ra
    
    for z in alkaline_earth:
        assert z in CLOSED_SHELL_ATOMS, (
            f"Alkaline earth metal Z={z} should be in CLOSED_SHELL_ATOMS"
        )
        atom = Atom(atomic_number=z)
        assert atom.open_shell is False, (
            f"Alkaline earth metal Z={z} should have open_shell=False"
        )


# ==============================================================================
# Tests for charge and n_electrons on Atom
# ==============================================================================


CHARGE_VALUES = [+2, +1, 0, -1, -2]


@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_atom_charge_stored(charge: int) -> None:
    """Test that the charge attribute is stored correctly on the Atom."""
    atom = Atom(atomic_number=8, charge=charge)
    assert atom.charge == charge


@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_atom_n_electrons_with_charge(charge: int) -> None:
    """Test n_electrons = Z + q for a single atom across charge values.

    :param charge: formal charge on the atom.
    """
    z = 8  # Oxygen
    atom = Atom(atomic_number=z, charge=charge)
    assert atom.n_electrons == z + charge


@pytest.mark.parametrize("atomic_number", [1, 6, 8, 26, 79])
@pytest.mark.parametrize("charge", CHARGE_VALUES)
def test_atom_n_electrons_various_elements(
    atomic_number: int, charge: int
) -> None:
    """Test n_electrons = Z + q for several elements and all charge values."""
    atom = Atom(atomic_number=atomic_number, charge=charge)
    assert atom.n_electrons == atomic_number + charge


def test_atom_default_charge_is_zero() -> None:
    """Test that the default charge is 0 (neutral atom)."""
    atom = Atom(atomic_number=6)
    assert atom.charge == 0
    assert atom.n_electrons == 6


def test_atom_n_electrons_neutral_equals_atomic_number() -> None:
    """For a neutral atom, n_electrons must equal Z for all elements."""
    for z in range(1, 119):
        atom = Atom(atomic_number=z)
        assert atom.n_electrons == z, (
            f"Neutral Atom(Z={z}) should have n_electrons={z}, "
            f"got {atom.n_electrons}"
        )


def test_atom_cation_fewer_electrons() -> None:
    """A cation (negative charge in our convention) should have fewer electrons."""
    # Na+ (sodium cation): Z=11, charge=-1 → N=10
    atom = Atom(atomic_number=11, charge=-1)
    assert atom.n_electrons == 10


def test_atom_anion_more_electrons() -> None:
    """An anion (positive charge in our convention) should have more electrons."""
    # Cl- (chloride): Z=17, charge=+1 → N=18
    atom = Atom(atomic_number=17, charge=+1)
    assert atom.n_electrons == 18


