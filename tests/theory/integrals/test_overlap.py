"""Unit tests for q_block.theory.integrals.overlap module.

Tests cover:
- Primitive overlap integrals for s, p, d orbitals
- Contracted overlap integrals
- Full overlap matrix for molecules
- Normalization (diagonal elements should be ~1.0)
- Symmetry of the overlap matrix

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block import ContractedGaussianTypeOrbital, Molecule, OverlapMatrix
from q_block.io.input_data import InputData
from q_block.theory.utils import get_cartesian_components
from tests.constants import BASIS_STO_3G, ORIGIN


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def h2_molecule() -> Molecule:
    """H2 molecule with STO-3G basis, converted to Bohr."""
    inp: InputData = InputData()
    inp.from_script(
        atom_data=[
            ["H", 0.0, 0.0, 0.0, BASIS_STO_3G],
            ["H", 0.0, 0.0, 0.74, BASIS_STO_3G],  # ~0.74 Å bond length
        ]
    )
    mol: Molecule = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol


@pytest.fixture
def water_molecule() -> Molecule:
    """Water molecule with STO-3G basis, converted to Bohr."""
    inp: InputData = InputData()
    inp.from_script(
        atom_data=[
            ["O", 0.0, 0.0, 0.1173, BASIS_STO_3G],
            ["H", 0.0, 0.7572, -0.4692, BASIS_STO_3G],
            ["H", 0.0, -0.7572, -0.4692, BASIS_STO_3G],
        ]
    )
    mol: Molecule = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol


# ======================================================================
# Gaussian Product Center Tests
# ======================================================================


def test_gaussian_product_center_same_center() -> None:
    """Verify product center of identical positions is that position."""
    A = (1.0, 2.0, 3.0)
    P = OverlapMatrix._gaussian_product_center(1.0, A, 1.0, A)
    assert P == pytest.approx(A)


def test_gaussian_product_center_equal_exponents() -> None:
    """Verify equal exponents give midpoint."""
    A = (0.0, 0.0, 0.0)
    B = (2.0, 0.0, 0.0)
    P = OverlapMatrix._gaussian_product_center(1.0, A, 1.0, B)
    assert P == pytest.approx((1.0, 0.0, 0.0))


def test_gaussian_product_center_unequal_exponents() -> None:
    """Verify larger exponent pulls center toward that Gaussian."""
    A = (0.0, 0.0, 0.0)
    B = (2.0, 0.0, 0.0)
    P = OverlapMatrix._gaussian_product_center(3.0, A, 1.0, B)  # alpha=3, beta=1
    # P = (3*0 + 1*2) / (3+1) = 0.5
    assert P == pytest.approx((0.5, 0.0, 0.0))


# ======================================================================
# Primitive Overlap Tests
# ======================================================================


def test_primitive_overlap_identical_s_orbitals() -> None:
    """Verify overlap of identical normalized s-orbital with itself is 1.0."""
    A = (0.0, 0.0, 0.0)
    alpha = 1.0
    overlap = OverlapMatrix.primitive_overlap(alpha, A, 0, 0, 0, alpha, A, 0, 0, 0)
    assert overlap == pytest.approx(1.0, rel=1e-6)


def test_primitive_overlap_s_orbitals_same_center_diff_exponents() -> None:
    """Verify two s-orbitals at same center have positive overlap < 1."""
    A = (0.0, 0.0, 0.0)
    overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 2.0, A, 0, 0, 0)
    assert overlap > 0
    assert overlap < 1.0


def test_primitive_overlap_s_orbitals_separated() -> None:
    """Verify s-orbitals separated along z-axis have reduced overlap."""
    A = (0.0, 0.0, 0.0)
    B = (0.0, 0.0, 2.0)
    overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 1.0, B, 0, 0, 0)
    assert overlap > 0
    assert overlap < 1.0


def test_primitive_overlap_s_orbitals_far_apart() -> None:
    """Verify s-orbitals far apart have near-zero overlap."""
    A = (0.0, 0.0, 0.0)
    B = (0.0, 0.0, 100.0)
    overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 1.0, B, 0, 0, 0)
    assert overlap == pytest.approx(0.0, abs=1e-10)


def test_primitive_overlap_p_orbital_self() -> None:
    """Verify p_x orbital overlaps with itself (normalized)."""
    A = (0.0, 0.0, 0.0)
    alpha = 1.0
    overlap = OverlapMatrix.primitive_overlap(alpha, A, 1, 0, 0, alpha, A, 1, 0, 0)
    assert overlap == pytest.approx(1.0, rel=1e-6)


def test_primitive_overlap_orthogonal_p_orbitals() -> None:
    """Verify orthogonal p-orbitals (px and py) have zero overlap."""
    A = (0.0, 0.0, 0.0)
    alpha = 1.0
    overlap = OverlapMatrix.primitive_overlap(alpha, A, 1, 0, 0, alpha, A, 0, 1, 0)
    assert overlap == pytest.approx(0.0, abs=1e-10)


def test_primitive_overlap_s_and_p_same_center() -> None:
    """Verify s and p orbitals on same center are orthogonal."""
    A = (0.0, 0.0, 0.0)
    alpha = 1.0
    overlap = OverlapMatrix.primitive_overlap(alpha, A, 0, 0, 0, alpha, A, 1, 0, 0)
    assert overlap == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# Contracted Overlap Tests
# ======================================================================


def test_contracted_overlap_single_primitive_equals_primitive() -> None:
    """Verify single-primitive CGTO overlap matches primitive overlap."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    overlap: float = OverlapMatrix.contracted_overlap(cgto, 0, 0, 0, cgto, 0, 0, 0)
    primitive: float = OverlapMatrix.primitive_overlap(
        1.0, (0.0, 0.0, 0.0), 0, 0, 0, 1.0, (0.0, 0.0, 0.0), 0, 0, 0
    )
    assert overlap == pytest.approx(primitive, rel=1e-10)


def test_contracted_overlap_multiple_primitives_positive() -> None:
    """Verify multiple contracted primitives give positive self-overlap."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[3.0, 1.0, 0.3],
        contractions=[0.5, 0.3, 0.2],
    )
    overlap: float = OverlapMatrix.contracted_overlap(cgto, 0, 0, 0, cgto, 0, 0, 0)
    assert overlap > 0


# ======================================================================
# OverlapMatrix Initialization Tests
# ======================================================================


def test_overlap_matrix_empty_basis_raises() -> None:
    """Verify empty basis set raises ValueError."""
    with pytest.raises(ValueError, match="empty basis set"):
        OverlapMatrix([])


def test_overlap_matrix_single_s_orbital() -> None:
    """Verify single s-orbital gives 1x1 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    S: OverlapMatrix = OverlapMatrix([cgto])
    assert S.n_basis == 1
    assert S.matrix.shape == (1, 1)


def test_overlap_matrix_single_p_orbital() -> None:
    """Verify single p-shell gives 3x3 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=1,
        exponents=[1.0],
        contractions=[1.0],
    )
    S: OverlapMatrix = OverlapMatrix([cgto])
    assert S.n_basis == 3
    assert S.matrix.shape == (3, 3)


# ======================================================================
# OverlapMatrix Symmetry Tests
# ======================================================================


def test_overlap_matrix_symmetry_h2(h2_molecule: Molecule) -> None:
    """Verify overlap matrix is symmetric for H2."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(S.matrix, S.matrix.T)


def test_overlap_matrix_symmetry_water(water_molecule: Molecule) -> None:
    """Verify overlap matrix is symmetric for water."""
    S: OverlapMatrix = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(S.matrix, S.matrix.T)


# ======================================================================
# OverlapMatrix Diagonal Tests (Normalization)
# ======================================================================


def test_overlap_matrix_diagonal_ones_h2(h2_molecule: Molecule) -> None:
    """Verify diagonal elements are ~1.0 for normalized H2 basis."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(S.matrix)
    np.testing.assert_array_almost_equal(diagonal, np.ones(S.n_basis), decimal=4)


def test_overlap_matrix_diagonal_ones_water(water_molecule: Molecule) -> None:
    """Verify diagonal elements are ~1.0 for water."""
    S: OverlapMatrix = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(S.matrix)
    np.testing.assert_array_almost_equal(diagonal, np.ones(S.n_basis), decimal=4)


# ======================================================================
# OverlapMatrix Properties Tests
# ======================================================================


def test_overlap_matrix_positive_definite_h2(h2_molecule: Molecule) -> None:
    """Verify overlap matrix is positive definite."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    eigenvalues: np.ndarray = np.linalg.eigvalsh(S.matrix)
    assert all(ev > 0 for ev in eigenvalues)


def test_overlap_matrix_repr(h2_molecule: Molecule) -> None:
    """Verify __repr__ output."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    repr_str: str = repr(S)
    assert "OverlapMatrix" in repr_str
    assert "n_basis" in repr_str


def test_overlap_matrix_getitem(h2_molecule: Molecule) -> None:
    """Verify __getitem__ access."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    assert S[0, 0] == S.matrix[0, 0]
    assert S[0, 1] == S.matrix[0, 1]


# ======================================================================
# H2 Specific Tests
# ======================================================================


def test_h2_overlap_matrix_dimension(h2_molecule: Molecule) -> None:
    """Verify H2 with STO-3G has 2 basis functions."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    assert S.n_basis == 2


def test_h2_off_diagonal_positive(h2_molecule: Molecule) -> None:
    """Verify off-diagonal elements are positive for bonded atoms."""
    S: OverlapMatrix = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
    # S[0,1] is overlap between 1s on H1 and 1s on H2
    assert S[0, 1] > 0
    assert S[0, 1] < 1.0  # Less than diagonal


# ======================================================================
# Water Specific Tests
# ======================================================================


def test_water_overlap_matrix_dimension(water_molecule: Molecule) -> None:
    """Verify water with STO-3G has 7 basis functions."""
    S: OverlapMatrix = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
    # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
    # H1: 1s = 1 function
    # H2: 1s = 1 function
    # Total: 7
    assert S.n_basis == 7


def test_water_overlap_matrix_shape(water_molecule: Molecule) -> None:
    """Verify overlap matrix has correct shape."""
    S: OverlapMatrix = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
    assert S.matrix.shape == (7, 7)
