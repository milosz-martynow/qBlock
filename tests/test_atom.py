"""
Tests for the SpinOrbital class.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

from q_block.atom import Atom, Orbital, Shell, SpinOrbital, SubShell
from q_block.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)
from q_block.basis_set_pople import parse_gaussian_basis
from tests.verification_data.expected_atom_empirical import (
    EXPECTED_ATOM_EMPIRICAL,
)
from tests.verification_data.expected_atom_pure import EXPECTED_ATOM_PURE

SpinKey = Tuple[int, int, int, float]  # (n, l, m, s)
SpinMap = Dict[SpinKey, bool]

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")
GOLDEN_ROOT: Path = Path("./tests/verification_data/gto_population")

BASIS_FILES = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311++Gss.gbs",
]


def test_spinorbital_valid_initialization():
    """
    Test that SpinOrbital initializes correctly with valid quantum numbers.

    This test verifies that valid (n, l, m, s) combinations create an object
    without raising errors and that the repr contains the expected fields.

    :returns: None
    """
    so = SpinOrbital(n=2, l=1, m=0, s=0.5)
    assert so.n == 2
    assert so.l == 1
    assert so.m == 0
    assert so.s == 0.5
    assert so.occupied is False
    r = repr(so)
    assert "So(n=2" in r and "l=1" in r


@pytest.mark.parametrize(
    "n,l,m,s",
    [
        (0, 0, 0, 0.5),  # invalid n
        (2, 2, 0, 0.5),  # invalid l (l >= n)
        (2, 1, 2, 0.5),  # invalid m (|m| > l)
        (1, 0, 0, 0.0),  # invalid s
    ],
)
def test_spinorbital_invalid_parameters(n, l, m, s):
    """
    Test that SpinOrbital raises ValueError for invalid quantum numbers.

    Each parameter set should be rejected by the constructor.

    :param int n: principal quantum number
    :param int l: orbital angular momentum quantum number
    :param int m: magnetic quantum number
    :param float s: spin projection
    :returns: None
    """
    with pytest.raises(ValueError):
        SpinOrbital(n=n, l=l, m=m, s=s)


def test_orbital_pairing_valid():
    """
    Test that Orbital accepts spin-up and spin-down SpinOrbitals with matching
    (n, l, m) and that repr contains expected fields.

    :returns: None
    """
    up = SpinOrbital(n=3, l=1, m=-1, s=0.5)
    down = SpinOrbital(n=3, l=1, m=-1, s=-0.5)
    orb = Orbital(spin_up=up, spin_down=down)
    assert orb.n == 3
    assert orb.l == 1
    assert orb.m == -1
    assert orb.spin_up is up
    assert orb.spin_down is down
    assert "Orb(n=3" in repr(orb)


def test_orbital_spin_mismatch_raises():
    """
    Test that providing incorrect spins raises a ValueError.

    :returns: None
    """
    up = SpinOrbital(n=2, l=0, m=0, s=0.5)
    wrong_spin = SpinOrbital(n=2, l=0, m=0, s=0.5)  # also +0.5, should be -0.5
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=wrong_spin)


def test_orbital_quantum_mismatch_raises():
    """
    Test that providing SpinOrbitals with different (n, l, m) raises.

    :returns: None
    """
    up = SpinOrbital(n=2, l=1, m=0, s=0.5)
    down = SpinOrbital(n=3, l=1, m=0, s=-0.5)  # n differs
    with pytest.raises(ValueError):
        Orbital(spin_up=up, spin_down=down)


def test_subshell_orbital_count_and_capacity():
    """
    Verify that SubShell creates (2l + 1) spatial orbitals and capacity equals
    2 * (2l + 1).

    The test checks orbitals length and the capacity() convenience method.

    :returns: None
    """
    n = 4
    l = 2  # d-subshell -> m = -2,-1,0,1,2 => 5 spatial orbitals
    sub = SubShell(n=n, l=l)
    assert len(sub.orbitals) == (2 * l + 1)
    assert sub.capacity() == 2 * (2 * l + 1)

    # each orbital must have spin-up and spin-down SpinOrbitals with expected
    # quantum numbers
    ms = list(range(-l, l + 1))
    observed_ms = [orb.m for orb in sub.orbitals]
    assert observed_ms == ms


@pytest.mark.parametrize(
    "invalid",
    [
        (0, 0),  # n < 1
        (2, 2),  # l >= n
        (-1, 0),  # negative n
    ],
)
def test_subshell_invalid_params(invalid):
    """
    SubShell should raise ValueError for invalid (n, l) specifications.

    :param tuple invalid: (n, l)
    :returns: None
    """
    n, l = invalid
    with pytest.raises(ValueError):
        SubShell(n=n, l=l)


def test_shell_initialization_and_subshells():
    """
    Verify Shell creates subshells for l in 0..n-1 and each subshell
    has the correct n and l values.

    :returns: None
    """
    n = 5
    shell = Shell(n=n)
    assert shell.n == n
    # should create n subshells (l = 0..n-1)
    assert len(shell.subshells) == n
    for l, subshell in enumerate(shell.subshells):
        assert isinstance(subshell, SubShell)
        assert subshell.n == n
        assert subshell.l == l


def test_shell_invalid_n():
    """
    Shell should raise ValueError when initialized with n < 1.

    :returns: None
    """
    with pytest.raises(ValueError):
        Shell(n=0)


@pytest.fixture(params=range(1, 119))
def atomic_number(request) -> int:
    """
    Fixture: iterate across all atomic numbers Z = 1..118.

    :param request: pytest parameter provider
    :returns: atomic number Z
    :rtype: int
    """
    return int(request.param)


@pytest.fixture
def expected_pure(atomic_number: int) -> SpinMap:
    """
    Provide expected pure-mode spin-orbital map for Z.

    :param atomic_number: atomic number
    :returns: expected spin-orbital occupancy dictionary
    """
    return EXPECTED_ATOM_PURE[atomic_number]


@pytest.fixture
def expected_emp(atomic_number: int):
    """
    Provide expected empirical-mode spin-orbital map for Z *only if it exists*.

    If Z is not present in EXPECTED_ATOM_EMPIRICAL, return None, meaning
    empirical == pure for this atomic number.

    :param atomic_number: atomic number
    :returns: empirical spin map or None
    """
    return EXPECTED_ATOM_EMPIRICAL.get(atomic_number, None)


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


def test_atom_pure_matches_expected(
    atomic_number: int,
    expected_pure: SpinMap,
) -> None:
    """
    Test that Atom(Z, pure mode) matches the expected static spin map.

    :param atomic_number: atomic number under test
    :param expected_pure: reference mapping from EXPECTED_ATOM_PURE
    """
    atom = Atom(Z=atomic_number, n_max=7, use_empirical_exceptions=False)
    atom.fill_occupancy()

    actual = extract_spin_map(atom)

    assert set(actual.keys()) == set(
        expected_pure.keys()
    ), f"Spin-orbital key mismatch for Z={atomic_number}"

    for key in actual:
        assert (
            actual[key] == expected_pure[key]
        ), f"Value mismatch at Z={atomic_number} on spin orbital {key}"

    assert (
        count_spin_map(actual) == atomic_number
    ), f"Electron count mismatch for Z={atomic_number}"


def test_atom_pure_matches_expected_for_exceptions(
    atomic_number: int,
    expected_pure: SpinMap,
) -> None:
    """
    Test that Atom(Z, pure mode) matches the expected pure spin-orbital map,
    but **only for atoms that have empirical exceptions**.

    Meaning:
      - Z is tested only if it appears in EXPECTED_ATOM_EMPIRICAL
      - For other Z, this test is skipped

    :param atomic_number: atomic number under test
    :param expected_pure: reference pure-mode spin map
    """
    # Only test atoms that actually have empirical exceptions
    # if atomic_number not in EXPECTED_ATOM_EMPIRICAL:
    #     pytest.skip(f"Z={atomic_number} has no empirical exception — skipping pure test.")

    atom = Atom(Z=atomic_number, n_max=7, use_empirical_exceptions=False)
    atom.fill_occupancy()

    # Ensure the atom's state is correctly updated without relying on a return value
    actual = extract_spin_map(atom)

    # Keyset must match exactly
    assert set(actual.keys()) == set(
        expected_pure.keys()
    ), f"[PURE] Spin-orbital key mismatch for exception atom Z={atomic_number}"

    # Values must match exactly
    for key in actual:
        assert (
            actual[key] == expected_pure[key]
        ), f"[PURE] Value mismatch at Z={atomic_number} on spin orbital {key}"

    # Pure electron count must always equal Z
    assert (
        count_spin_map(actual) == atomic_number
    ), f"[PURE] Electron count mismatch for Z={atomic_number}"


"""
Golden-reference tests for GTO population.

===============================================================================
PURPOSE
===============================================================================

This test module verifies the correctness of:

    Atom.populate_spinorbitals_with_gto

by comparing its output against *golden-reference data* generated by:

    tools/generate_golden.py

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
        "symbol": ATOMS_SYMBOLS_Z_TO_SYMBOL[atom.Z],
        "Z": atom.Z,
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

    basis = parse_gaussian_basis(filepath=str(basis_path))
    golden_atom = golden["atoms"][symbol]

    atom = Atom(
        Z=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
        basis_set=basis[symbol],
    )
    atom.populate_spinorbitals_with_gto()

    snapshot = _serialize_atom_for_test(
        atom=atom,
        basis_name=basis_name,
    )

    assert snapshot == golden_atom, (
        f"GTO population mismatch for atom {symbol} " f"in basis {basis_name}"
    )
