"""Unit tests for q_block.theory.basis_functions module.

Tests cover:
- ContractedGaussianTypeOrbital class initialization, validation, and methods
- Atom.make_contracted_gaussian_type_orbital
- Molecule.make_contracted_gaussian_type_orbital
- Molecule.evaluate_molecular_orbital

All tests use pytest with parametrize, no test classes.
"""

from typing import List, Tuple

import pytest

from q_block import Atom, ContractedGaussianTypeOrbital, Molecule
from q_block.io.basis_set import Pople
from q_block.io.coordinates import CartesianCoordinates
from q_block.io.input_data import InputData
from q_block.theory.utils import normalization_constant
from tests.unit_tests.constants import BASIS_6_31G, ORIGIN


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_cgto() -> ContractedGaussianTypeOrbital:
    """Simple s-type CGTO with one primitive for testing."""
    return ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
        atom_index=0,
    )


@pytest.fixture
def hydrogen_atom() -> Atom:
    """Hydrogen atom at origin with 6-31G basis."""
    return Atom(
        atomic_number=1,
        basis_set=BASIS_6_31G,
        coordinates=ORIGIN,
    )


@pytest.fixture
def water_molecule() -> Molecule:
    """Water molecule with 6-31G basis."""
    inp: InputData = InputData()
    inp.from_script(
        atom_data=[
            ["O", 0.0, 0.0, 0.1173, BASIS_6_31G],
            ["H", 0.0, 0.7572, -0.4692, BASIS_6_31G],
            ["H", 0.0, -0.7572, -0.4692, BASIS_6_31G],
        ]
    )
    return Molecule(input_data=inp)


# ======================================================================
# CGTO Initialization Tests
# ======================================================================


@pytest.mark.parametrize(
    "l, expected_n_functions",
    [
        (0, 1),   # s-orbital: 2*0 + 1 = 1
        (1, 3),   # p-orbital: 2*1 + 1 = 3
        (2, 5),   # d-orbital: 2*2 + 1 = 5
        (3, 7),   # f-orbital: 2*3 + 1 = 7
    ],
    ids=["s_orbital", "p_orbital", "d_orbital", "f_orbital"],
)
def test_cgto_init_orbital_types(l: int, expected_n_functions: int) -> None:
    """Verify CGTO initialization for different angular momentum.

    :param l: Angular momentum quantum number.
    :type l: int
    :param expected_n_functions: Expected number of basis functions (2l+1).
    :type expected_n_functions: int
    """
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=l,
        exponents=[1.0],
        contractions=[1.0],
    )
    assert cgto.l == l
    assert cgto.n_functions == expected_n_functions


def test_cgto_init_basic() -> None:
    """Verify basic CGTO initialization with multiple primitives."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0, 0.5],
        contractions=[0.6, 0.4],
        atom_index=0,
    )
    assert cgto.center == ORIGIN
    assert cgto.l == 0
    assert cgto.exponents == [1.0, 0.5]
    assert cgto.contractions == [0.6, 0.4]
    assert cgto.atom_index == 0
    assert cgto.n_primitives == 2


def test_cgto_init_atom_index_none() -> None:
    """Verify atom_index defaults to None."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    assert cgto.atom_index is None


def test_cgto_init_length_mismatch_raises() -> None:
    """Verify mismatched exponents/contractions raises ValueError."""
    with pytest.raises(ValueError, match="same length"):
        ContractedGaussianTypeOrbital(
            center=ORIGIN,
            l=0,
            exponents=[1.0, 0.5],
            contractions=[1.0],
        )


def test_cgto_init_empty_exponents_raises() -> None:
    """Verify empty exponents raises ValueError."""
    with pytest.raises(ValueError, match="at least one primitive"):
        ContractedGaussianTypeOrbital(
            center=ORIGIN,
            l=0,
            exponents=[],
            contractions=[],
        )


# ======================================================================
# CGTO Dunder Methods Tests
# ======================================================================


@pytest.mark.parametrize(
    "l, expected_char",
    [
        (0, "s"),
        (1, "p"),
        (2, "d"),
    ],
    ids=["s", "p", "d"],
)
def test_cgto_repr_orbital_type(l: int, expected_char: str) -> None:
    """Verify __repr__ contains correct orbital type character.

    :param l: Angular momentum quantum number.
    :type l: int
    :param expected_char: Expected orbital type character in repr.
    :type expected_char: str
    """
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=l,
        exponents=[1.0],
        contractions=[1.0],
    )
    assert expected_char in repr(cgto)


def test_cgto_repr_s_orbital(simple_cgto: ContractedGaussianTypeOrbital) -> None:
    """Verify __repr__ for s-orbital contains expected fields."""
    repr_str: str = repr(simple_cgto)
    assert "CGTO" in repr_str
    assert "atom=0" in repr_str
    assert "s" in repr_str
    assert "K=1" in repr_str


def test_cgto_eq_same() -> None:
    """Verify equality of identical CGTOs."""
    cgto1: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
        atom_index=0,
    )
    cgto2: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
        atom_index=0,
    )
    assert cgto1 == cgto2


def test_cgto_eq_different_l() -> None:
    """Verify inequality when l differs."""
    cgto1: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN, l=0, exponents=[1.0], contractions=[1.0]
    )
    cgto2: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN, l=1, exponents=[1.0], contractions=[1.0]
    )
    assert cgto1 != cgto2


def test_cgto_eq_different_exponents() -> None:
    """Verify inequality when exponents differ."""
    cgto1: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN, l=0, exponents=[1.0], contractions=[1.0]
    )
    cgto2: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN, l=0, exponents=[2.0], contractions=[1.0]
    )
    assert cgto1 != cgto2


def test_cgto_eq_not_cgto(simple_cgto: ContractedGaussianTypeOrbital) -> None:
    """Verify comparison with non-CGTO returns NotImplemented."""
    assert simple_cgto.__eq__("not a cgto") == NotImplemented


# ======================================================================
# CGTO Evaluation Tests
# ======================================================================


def test_primitive_gaussian_at_center(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify primitive Gaussian at center equals normalization constant."""
    value: float = simple_cgto.evaluate_primitive_gaussian(ORIGIN, 0, 0, 0, 0)
    expected: float = normalization_constant(1.0, 0, 0, 0)
    assert abs(value - expected) < 1e-10


def test_primitive_gaussian_decays(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify primitive Gaussian decays away from center."""
    val_0: float = simple_cgto.evaluate_primitive_gaussian(ORIGIN, 0, 0, 0, 0)
    val_1: float = simple_cgto.evaluate_primitive_gaussian(
        CartesianCoordinates(1.0, 0.0, 0.0), 0, 0, 0, 0
    )
    val_2: float = simple_cgto.evaluate_primitive_gaussian(
        CartesianCoordinates(2.0, 0.0, 0.0), 0, 0, 0, 0
    )
    assert val_0 > val_1 > val_2 > 0


def test_primitive_gaussian_index_error(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify primitive_gaussian raises IndexError for invalid index."""
    with pytest.raises(IndexError, match="out of range"):
        simple_cgto.evaluate_primitive_gaussian(ORIGIN, 5, 0, 0, 0)
    with pytest.raises(IndexError, match="out of range"):
        simple_cgto.evaluate_primitive_gaussian(ORIGIN, -1, 0, 0, 0)


def test_primitive_gaussian_angular_mismatch(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify primitive_gaussian raises ValueError when lx+ly+lz != l."""
    with pytest.raises(ValueError, match="must equal l"):
        simple_cgto.evaluate_primitive_gaussian(ORIGIN, 0, 1, 0, 0)


def test_primitive_gaussian_p_orbital() -> None:
    """Verify primitive Gaussian for p-orbital at origin is zero."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=1,
        exponents=[1.0],
        contractions=[1.0],
    )
    # p_x at center: x^1 * ... = 0
    assert cgto.evaluate_primitive_gaussian(ORIGIN, 0, 1, 0, 0) == 0.0

    # p_x off-center: should be non-zero
    off_center: CartesianCoordinates = CartesianCoordinates(0.5, 0.0, 0.0)
    assert cgto.evaluate_primitive_gaussian(off_center, 0, 1, 0, 0) != 0.0


def test_basis_function_s_orbital(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify basis_function for single-primitive s-orbital."""
    bf_value: float = simple_cgto.evaluate_basis_function(ORIGIN, 0, 0, 0)
    prim_value: float = simple_cgto.evaluate_primitive_gaussian(ORIGIN, 0, 0, 0, 0)
    # With contraction coefficient of 1.0, they should be equal
    assert abs(bf_value - 1.0 * prim_value) < 1e-10


def test_basis_function_angular_mismatch(
    simple_cgto: ContractedGaussianTypeOrbital,
) -> None:
    """Verify basis_function raises ValueError when lx+ly+lz != l."""
    with pytest.raises(ValueError, match="must equal l"):
        simple_cgto.evaluate_basis_function(ORIGIN, 1, 0, 0)


def test_basis_function_contracted() -> None:
    """Verify basis_function correctly sums over primitives."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0, 0.5],
        contractions=[0.6, 0.4],
    )
    bf_value: float = cgto.evaluate_basis_function(ORIGIN, 0, 0, 0)

    prim0: float = cgto.evaluate_primitive_gaussian(ORIGIN, 0, 0, 0, 0)
    prim1: float = cgto.evaluate_primitive_gaussian(ORIGIN, 1, 0, 0, 0)
    expected: float = 0.6 * prim0 + 0.4 * prim1
    assert abs(bf_value - expected) < 1e-10


# ======================================================================
# Atom.make_contracted_gaussian_type_orbital Tests
# ======================================================================


def test_atom_make_cgto_populates_list(hydrogen_atom: Atom) -> None:
    """Verify method populates contracted_gaussian_type_orbitals."""
    assert hydrogen_atom.contracted_gaussian_type_orbitals is None
    hydrogen_atom.make_contracted_gaussian_type_orbital()
    assert hydrogen_atom.contracted_gaussian_type_orbitals is not None
    assert len(hydrogen_atom.contracted_gaussian_type_orbitals) > 0


def test_atom_make_cgto_returns_none(hydrogen_atom: Atom) -> None:
    """Verify method returns None."""
    result = hydrogen_atom.make_contracted_gaussian_type_orbital()
    assert result is None


def test_atom_make_cgto_atom_index_none(hydrogen_atom: Atom) -> None:
    """Verify atom_index is None when not provided."""
    hydrogen_atom.make_contracted_gaussian_type_orbital()
    for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
        assert cgto.atom_index is None


def test_atom_make_cgto_atom_index_set(hydrogen_atom: Atom) -> None:
    """Verify atom_index is set when provided."""
    hydrogen_atom.make_contracted_gaussian_type_orbital(atom_index=5)
    for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
        assert cgto.atom_index == 5


def test_atom_make_cgto_missing_basis_set() -> None:
    """Verify method handles missing basis set gracefully."""
    atom: Atom = Atom(atomic_number=1, coordinates=ORIGIN)
    atom.make_contracted_gaussian_type_orbital()
    assert atom.contracted_gaussian_type_orbitals == []


def test_atom_make_cgto_missing_coordinates() -> None:
    """Verify method handles missing coordinates gracefully."""
    atom: Atom = Atom(atomic_number=1, basis_set=BASIS_6_31G)
    atom.make_contracted_gaussian_type_orbital()
    assert atom.contracted_gaussian_type_orbitals == []


def test_atom_make_cgto_missing_both() -> None:
    """Verify method handles missing basis set and coordinates."""
    atom: Atom = Atom(atomic_number=1)
    atom.make_contracted_gaussian_type_orbital()
    assert atom.contracted_gaussian_type_orbitals == []


def test_atom_make_cgto_center_is_cartesian(hydrogen_atom: Atom) -> None:
    """Verify CGTO center is CartesianCoordinates instance."""
    hydrogen_atom.make_contracted_gaussian_type_orbital()
    for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
        assert isinstance(cgto.center, CartesianCoordinates)


def test_atom_make_cgto_hydrogen_631g_shell_count(hydrogen_atom: Atom) -> None:
    """Verify hydrogen with 6-31G has expected number of shells."""
    hydrogen_atom.make_contracted_gaussian_type_orbital()
    # 6-31G for H: 1s (3 primitives) + 1s' (1 primitive) = 2 shells
    assert len(hydrogen_atom.contracted_gaussian_type_orbitals) == 2


def test_atom_make_cgto_oxygen_631g_shell_count() -> None:
    """Verify oxygen with 6-31G has expected number of shells."""
    oxygen: Atom = Atom(atomic_number=8, basis_set=BASIS_6_31G, coordinates=ORIGIN)
    oxygen.make_contracted_gaussian_type_orbital()
    # 6-31G for O: 1s + 2s + 2p + 2s' + 2p' = 5 shells
    assert len(oxygen.contracted_gaussian_type_orbitals) == 5


# ======================================================================
# Molecule.make_contracted_gaussian_type_orbital Tests
# ======================================================================


def test_molecule_make_cgto_populates_list(water_molecule: Molecule) -> None:
    """Verify method populates contracted_gaussian_type_orbitals."""
    assert water_molecule.contracted_gaussian_type_orbitals is None
    water_molecule.make_contracted_gaussian_type_orbital()
    assert water_molecule.contracted_gaussian_type_orbitals is not None
    assert len(water_molecule.contracted_gaussian_type_orbitals) > 0


def test_molecule_make_cgto_returns_none(water_molecule: Molecule) -> None:
    """Verify method returns None."""
    result = water_molecule.make_contracted_gaussian_type_orbital()
    assert result is None


def test_molecule_make_cgto_n_basis_matches_sum(water_molecule: Molecule) -> None:
    """Verify n_basis equals sum of n_functions over all CGTOs."""
    water_molecule.make_contracted_gaussian_type_orbital()
    total_functions: int = sum(
        cgto.n_functions for cgto in water_molecule.contracted_gaussian_type_orbitals
    )
    assert total_functions == water_molecule.n_basis


def test_molecule_make_cgto_atom_indices_correct(water_molecule: Molecule) -> None:
    """Verify each CGTO has correct atom_index."""
    water_molecule.make_contracted_gaussian_type_orbital()
    # Water: O (index 0), H (index 1), H (index 2)
    seen_indices: set = set()
    for cgto in water_molecule.contracted_gaussian_type_orbitals:
        seen_indices.add(cgto.atom_index)
    assert seen_indices == {0, 1, 2}


def test_molecule_make_cgto_water_631g_shell_count(water_molecule: Molecule) -> None:
    """Verify water with 6-31G has expected number of shells."""
    water_molecule.make_contracted_gaussian_type_orbital()
    # O: 5 shells, H: 2 shells each
    # Total: 5 + 2 + 2 = 9 shells
    assert len(water_molecule.contracted_gaussian_type_orbitals) == 9


def test_molecule_make_cgto_water_631g_n_basis(water_molecule: Molecule) -> None:
    """Verify water with 6-31G has 13 basis functions."""
    # O: 1s(1) + 2s(1) + 2p(3) + 2s'(1) + 2p'(3) = 9
    # H: 1s(1) + 1s'(1) = 2 each
    # Total: 9 + 2 + 2 = 13
    assert water_molecule.n_basis == 13


def test_molecule_make_cgto_also_populates_atoms(water_molecule: Molecule) -> None:
    """Verify method also populates CGTOs on individual atoms."""
    water_molecule.make_contracted_gaussian_type_orbital()
    for atom in water_molecule.atoms:
        assert atom.contracted_gaussian_type_orbitals is not None
        assert len(atom.contracted_gaussian_type_orbitals) > 0


# ======================================================================
# Molecule.molecular_orbital Tests
# ======================================================================


def _cartesian_components(l: int) -> List[Tuple[int, int, int]]:
    """Generate Cartesian angular momentum tuples for given l."""
    components: List[Tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            components.append((lx, ly, lz))
    return components


def _build_angular_components(
    cgtos: List[ContractedGaussianTypeOrbital],
) -> List[Tuple[int, int, int]]:
    """Build angular components list for all CGTOs."""
    angular: List[Tuple[int, int, int]] = []
    for cgto in cgtos:
        angular.extend(_cartesian_components(cgto.l))
    return angular


def test_molecular_orbital_raises_if_cgtos_not_built(
    water_molecule: Molecule,
) -> None:
    """Verify method raises if CGTOs haven't been built."""
    r: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.0)
    coeffs: List[float] = [1.0] * water_molecule.n_basis
    angular: List[Tuple[int, int, int]] = [(0, 0, 0)] * water_molecule.n_basis
    with pytest.raises(ValueError, match="CGTOs not built"):
        water_molecule.evaluate_molecular_orbital(r, coeffs, angular)


def test_molecular_orbital_raises_on_coefficients_mismatch(
    water_molecule: Molecule,
) -> None:
    """Verify method raises on coefficient length mismatch."""
    water_molecule.make_contracted_gaussian_type_orbital()
    r: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.0)
    angular: List[Tuple[int, int, int]] = _build_angular_components(
        water_molecule.contracted_gaussian_type_orbitals
    )
    coeffs: List[float] = [1.0]  # Too short
    with pytest.raises(ValueError, match="coefficients length"):
        water_molecule.evaluate_molecular_orbital(r, coeffs, angular)


def test_molecular_orbital_raises_on_angular_mismatch(
    water_molecule: Molecule,
) -> None:
    """Verify method raises on angular_components length mismatch."""
    water_molecule.make_contracted_gaussian_type_orbital()
    r: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.0)
    coeffs: List[float] = [1.0] * water_molecule.n_basis
    angular: List[Tuple[int, int, int]] = [(0, 0, 0)]  # Too short
    with pytest.raises(ValueError, match="angular_components length"):
        water_molecule.evaluate_molecular_orbital(r, coeffs, angular)


def test_molecular_orbital_unit_vector_first_basis(
    water_molecule: Molecule,
) -> None:
    """Verify MO evaluation with unit vector selecting first basis function."""
    water_molecule.make_contracted_gaussian_type_orbital()
    r: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.1173)  # Near oxygen
    angular: List[Tuple[int, int, int]] = _build_angular_components(
        water_molecule.contracted_gaussian_type_orbitals
    )
    coeffs: List[float] = [1.0] + [0.0] * (water_molecule.n_basis - 1)

    mo_value: float = water_molecule.evaluate_molecular_orbital(r, coeffs, angular)

    first_cgto: ContractedGaussianTypeOrbital = (
        water_molecule.contracted_gaussian_type_orbitals[0]
    )
    first_angular: Tuple[int, int, int] = angular[0]
    expected: float = first_cgto.evaluate_basis_function(r, *first_angular)
    assert abs(mo_value - expected) < 1e-10


def test_molecular_orbital_zero_coefficients_gives_zero(
    water_molecule: Molecule,
) -> None:
    """Verify all-zero coefficients give zero MO value."""
    water_molecule.make_contracted_gaussian_type_orbital()
    r: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.0)
    angular: List[Tuple[int, int, int]] = _build_angular_components(
        water_molecule.contracted_gaussian_type_orbitals
    )
    coeffs: List[float] = [0.0] * water_molecule.n_basis

    mo_value: float = water_molecule.evaluate_molecular_orbital(r, coeffs, angular)
    assert mo_value == 0.0


def test_molecular_orbital_linearity(water_molecule: Molecule) -> None:
    """Verify MO evaluation is linear in coefficients."""
    water_molecule.make_contracted_gaussian_type_orbital()
    r: CartesianCoordinates = CartesianCoordinates(0.5, 0.3, 0.2)
    angular: List[Tuple[int, int, int]] = _build_angular_components(
        water_molecule.contracted_gaussian_type_orbitals
    )

    # Compute with coefficients a and b
    coeffs_a: List[float] = [1.0] + [0.0] * (water_molecule.n_basis - 1)
    coeffs_b: List[float] = [0.0, 1.0] + [0.0] * (water_molecule.n_basis - 2)

    mo_a: float = water_molecule.evaluate_molecular_orbital(r, coeffs_a, angular)
    mo_b: float = water_molecule.evaluate_molecular_orbital(r, coeffs_b, angular)

    # Sum should equal evaluation with summed coefficients
    coeffs_sum: List[float] = [1.0, 1.0] + [0.0] * (water_molecule.n_basis - 2)
    mo_sum: float = water_molecule.evaluate_molecular_orbital(r, coeffs_sum, angular)

    assert abs(mo_sum - (mo_a + mo_b)) < 1e-10
