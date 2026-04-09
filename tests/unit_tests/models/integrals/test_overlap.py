"""Unit tests for q_block.models.integrals.overlap module.

Tests cover:
- Primitive overlap integrals for s, p, d orbitals
- Contracted overlap integrals
- Full overlap matrix for molecules
- Normalization (diagonal elements should be ~1.0)
- Symmetry of the overlap matrix

All tests use pytest with parametrize, no test classes.

TODO: 
- Add tests for specific molecules (e.g., H2, H2O) to verify expected overlap values
- add tests of overlap of orbitals higher that d (e.g., f, g) if implemented in the future
"""

import numpy as np
import pytest

from q_block import ContractedGaussianTypeOrbital, Molecule, Overlap
from tests.unit_tests.constants import ORIGIN
from tests.unit_tests.utils import (
    ALL_ORBITAL_COMPONENTS,
    get_orbital_ids,
    h2_molecule,
    water_molecule,
)


# ======================================================================
# Gaussian Product Center Tests
# ======================================================================


def test_gaussian_product_center_same_center() -> None:
    """Verify product center of identical positions is that position."""
    A: tuple[float, float, float] = (1.0, 2.0, 3.0)
    P: tuple[float, float, float] = Overlap._gaussian_product_center(
        alpha=1.0, A=A, beta=1.0, B=A
    )
    assert P == pytest.approx(expected=A)


def test_gaussian_product_center_equal_exponents() -> None:
    """Verify equal exponents give midpoint."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (2.0, 0.0, 0.0)
    P: tuple[float, float, float] = Overlap._gaussian_product_center(
        alpha=1.0, A=A, beta=1.0, B=B
    )
    assert P == pytest.approx(expected=(1.0, 0.0, 0.0))


def test_gaussian_product_center_unequal_exponents() -> None:
    """Verify larger exponent pulls center toward that Gaussian."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (2.0, 0.0, 0.0)
    P: tuple[float, float, float] = Overlap._gaussian_product_center(
        alpha=3.0, A=A, beta=1.0, B=B
    )
    # P = (3*0 + 1*2) / (3+1) = 0.5
    assert P == pytest.approx(expected=(0.5, 0.0, 0.0))


# ======================================================================
# Primitive Overlap Tests - Parametrized by Orbital Type
# ======================================================================


# ----------------------------------------------------------------------
# Self-Overlap Tests (normalized orbital with itself = 1.0)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_overlap_self(lx: int, ly: int, lz: int) -> None:
    """Verify overlap of identical normalized orbital with itself is 1.0.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    overlap: float = Overlap.primitive_overlap(
        alpha=alpha, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=alpha, B=A, lx2=lx, ly2=ly, lz2=lz
    )
    assert overlap == pytest.approx(expected=1.0, rel=1e-6)


# ----------------------------------------------------------------------
# Same Center, Different Exponents Tests (positive overlap < 1)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_overlap_same_center_diff_exponents(lx: int, ly: int, lz: int) -> None:
    """Verify two orbitals at same center with different exponents have positive overlap < 1.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    overlap: float = Overlap.primitive_overlap(
        alpha=1.0, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=2.0, B=A, lx2=lx, ly2=ly, lz2=lz
    )
    assert overlap > 0
    assert overlap < 1.0


# ----------------------------------------------------------------------
# Separated Orbitals Tests (reduced overlap magnitude)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_overlap_separated(lx: int, ly: int, lz: int) -> None:
    """Verify orbitals separated along z-axis have reduced overlap magnitude.

    Note: Overlap can be negative for orbitals oriented along the separation
    axis (e.g., pz separated along z) due to the nodal structure of the
    wave function.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (0.0, 0.0, 2.0)
    overlap: float = Overlap.primitive_overlap(
        alpha=1.0, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=1.0, B=B, lx2=lx, ly2=ly, lz2=lz
    )
    # Magnitude of overlap should be less than self-overlap (1.0)
    assert abs(overlap) < 1.0


# ----------------------------------------------------------------------
# Far Apart Orbitals Tests (near-zero overlap)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_overlap_far_apart(lx: int, ly: int, lz: int) -> None:
    """Verify orbitals far apart have near-zero overlap.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (0.0, 0.0, 100.0)
    overlap: float = Overlap.primitive_overlap(
        alpha=1.0, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=1.0, B=B, lx2=lx, ly2=ly, lz2=lz
    )
    assert overlap == pytest.approx(expected=0.0, abs=1e-10)


# ----------------------------------------------------------------------
# Orthogonality Tests - Different Components of Same Shell
# ----------------------------------------------------------------------

# Pairs of orthogonal p-orbitals
P_ORTHOGONAL_PAIRS = [
    ((1, 0, 0), (0, 1, 0), "px_py"),
    ((1, 0, 0), (0, 0, 1), "px_pz"),
    ((0, 1, 0), (0, 0, 1), "py_pz"),
]

# Pairs of orthogonal d-orbitals (different angular momentum components)
# Note: In Cartesian basis, dxx, dyy, dzz are NOT orthogonal to each other
# because they share s-like character (x² + y² + z² = r²).
# Only "pure" angular momentum orbitals (dxy, dxz, dyz) are orthogonal to
# the diagonal ones and to each other.
D_ORTHOGONAL_PAIRS = [
    ((2, 0, 0), (1, 1, 0), "dxx_dxy"),
    ((2, 0, 0), (1, 0, 1), "dxx_dxz"),
    ((2, 0, 0), (0, 1, 1), "dxx_dyz"),
    ((1, 1, 0), (1, 0, 1), "dxy_dxz"),
    ((1, 1, 0), (0, 2, 0), "dxy_dyy"),
    ((1, 1, 0), (0, 1, 1), "dxy_dyz"),
    ((1, 1, 0), (0, 0, 2), "dxy_dzz"),
    ((1, 0, 1), (0, 2, 0), "dxz_dyy"),
    ((1, 0, 1), (0, 1, 1), "dxz_dyz"),
    ((1, 0, 1), (0, 0, 2), "dxz_dzz"),
    ((0, 2, 0), (0, 1, 1), "dyy_dyz"),
    ((0, 1, 1), (0, 0, 2), "dyz_dzz"),
]

ALL_SAME_SHELL_ORTHOGONAL_PAIRS = P_ORTHOGONAL_PAIRS + D_ORTHOGONAL_PAIRS


@pytest.mark.parametrize(
    "orbital1, orbital2",
    [(p[0], p[1]) for p in ALL_SAME_SHELL_ORTHOGONAL_PAIRS],
    ids=[p[2] for p in ALL_SAME_SHELL_ORTHOGONAL_PAIRS],
)
def test_primitive_overlap_orthogonal_same_shell(
    orbital1: tuple[int, int, int], orbital2: tuple[int, int, int]
) -> None:
    """Verify orthogonal orbitals of same shell have zero overlap at same center.

    :param orbital1: tuple[int, int, int] - First orbital angular momentum (lx, ly, lz).
    :param orbital2: tuple[int, int, int] - Second orbital angular momentum (lx, ly, lz).
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    lx1, ly1, lz1 = orbital1
    lx2, ly2, lz2 = orbital2
    overlap: float = Overlap.primitive_overlap(
        alpha=alpha, A=A, lx1=lx1, ly1=ly1, lz1=lz1,
        beta=alpha, B=A, lx2=lx2, ly2=ly2, lz2=lz2
    )
    assert overlap == pytest.approx(expected=0.0, abs=1e-10)


# ----------------------------------------------------------------------
# Orthogonality Tests - Different Shells at Same Center
# ----------------------------------------------------------------------

# s with p orbitals (all orthogonal due to different parity)
S_P_PAIRS = [
    ((0, 0, 0), (1, 0, 0), "s_px"),
    ((0, 0, 0), (0, 1, 0), "s_py"),
    ((0, 0, 0), (0, 0, 1), "s_pz"),
]

# s with d orbitals
# Note: s is orthogonal to dxy, dxz, dyz (odd parity in each coordinate pair),
# but NOT orthogonal to dxx, dyy, dzz (same even parity as s).
S_D_PAIRS = [
    ((0, 0, 0), (1, 1, 0), "s_dxy"),
    ((0, 0, 0), (1, 0, 1), "s_dxz"),
    ((0, 0, 0), (0, 1, 1), "s_dyz"),
]

# p with d orbitals (orthogonal when parity differs in each coordinate)
P_D_PAIRS = [
    ((1, 0, 0), (0, 2, 0), "px_dyy"),
    ((1, 0, 0), (0, 1, 1), "px_dyz"),
    ((1, 0, 0), (0, 0, 2), "px_dzz"),
    ((0, 1, 0), (2, 0, 0), "py_dxx"),
    ((0, 1, 0), (1, 0, 1), "py_dxz"),
    ((0, 1, 0), (0, 0, 2), "py_dzz"),
    ((0, 0, 1), (2, 0, 0), "pz_dxx"),
    ((0, 0, 1), (1, 1, 0), "pz_dxy"),
    ((0, 0, 1), (0, 2, 0), "pz_dyy"),
]

ALL_CROSS_SHELL_PAIRS = S_P_PAIRS + S_D_PAIRS + P_D_PAIRS


@pytest.mark.parametrize(
    "orbital1, orbital2",
    [(p[0], p[1]) for p in ALL_CROSS_SHELL_PAIRS],
    ids=[p[2] for p in ALL_CROSS_SHELL_PAIRS],
)
def test_primitive_overlap_orthogonal_different_shells(
    orbital1: tuple[int, int, int], orbital2: tuple[int, int, int]
) -> None:
    """Verify orbitals from different shells are orthogonal at same center.

    :param orbital1: tuple[int, int, int] - First orbital angular momentum (lx, ly, lz).
    :param orbital2: tuple[int, int, int] - Second orbital angular momentum (lx, ly, lz).
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    lx1, ly1, lz1 = orbital1
    lx2, ly2, lz2 = orbital2
    overlap: float = Overlap.primitive_overlap(
        alpha=alpha, A=A, lx1=lx1, ly1=ly1, lz1=lz1,
        beta=alpha, B=A, lx2=lx2, ly2=ly2, lz2=lz2
    )
    assert overlap == pytest.approx(expected=0.0, abs=1e-10)


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
    overlap: float = Overlap.contracted_overlap(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0
    )
    primitive: float = Overlap.primitive_overlap(
        alpha=1.0, A=(0.0, 0.0, 0.0), lx1=0, ly1=0, lz1=0,
        beta=1.0, B=(0.0, 0.0, 0.0), lx2=0, ly2=0, lz2=0
    )
    assert overlap == pytest.approx(expected=primitive, rel=1e-10)


def test_contracted_overlap_multiple_primitives_positive() -> None:
    """Verify multiple contracted primitives give positive self-overlap."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[3.0, 1.0, 0.3],
        contractions=[0.5, 0.3, 0.2],
    )
    overlap: float = Overlap.contracted_overlap(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0
    )
    assert overlap > 0


# ======================================================================
# Overlap Initialization Tests
# ======================================================================


def test_overlap_matrix_empty_basis_raises() -> None:
    """Verify empty basis set raises ValueError."""
    with pytest.raises(expected_exception=ValueError, match="empty basis set"):
        Overlap(cgtos=[])


def test_overlap_matrix_single_s_orbital() -> None:
    """Verify single s-orbital gives 1x1 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    S: Overlap = Overlap(cgtos=[cgto])
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
    S: Overlap = Overlap(cgtos=[cgto])
    assert S.n_basis == 3
    assert S.matrix.shape == (3, 3)


# ======================================================================
# Overlap Symmetry Tests
# ======================================================================


def test_overlap_matrix_symmetry_h2(h2_molecule: Molecule) -> None:
    """Verify overlap matrix is symmetric for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(actual=S.matrix, desired=S.matrix.T)


def test_overlap_matrix_symmetry_water(water_molecule: Molecule) -> None:
    """Verify overlap matrix is symmetric for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(actual=S.matrix, desired=S.matrix.T)


# ======================================================================
# Overlap Diagonal Tests (Normalization)
# ======================================================================


def test_overlap_matrix_diagonal_ones_h2(h2_molecule: Molecule) -> None:
    """Verify diagonal elements are ~1.0 for normalized H2 basis.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(v=S.matrix)
    np.testing.assert_array_almost_equal(
        actual=diagonal, desired=np.ones(shape=S.n_basis), decimal=4
    )


def test_overlap_matrix_diagonal_ones_water(water_molecule: Molecule) -> None:
    """Verify diagonal elements are ~1.0 for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(v=S.matrix)
    np.testing.assert_array_almost_equal(
        actual=diagonal, desired=np.ones(shape=S.n_basis), decimal=4
    )


# ======================================================================
# Overlap Properties Tests
# ======================================================================


def test_overlap_matrix_positive_definite_h2(h2_molecule: Molecule) -> None:
    """Verify overlap matrix is positive definite.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    eigenvalues: np.ndarray = np.linalg.eigvalsh(a=S.matrix)
    assert all(ev > 0 for ev in eigenvalues)


def test_overlap_matrix_repr(h2_molecule: Molecule) -> None:
    """Verify __repr__ output.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    repr_str: str = repr(S)
    assert "Overlap" in repr_str
    assert "n_basis" in repr_str


def test_overlap_matrix_getitem(h2_molecule: Molecule) -> None:
    """Verify __getitem__ access.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    assert S[0, 0] == S.matrix[0, 0]
    assert S[0, 1] == S.matrix[0, 1]


# ======================================================================
# H2 Specific Tests
# ======================================================================


def test_h2_overlap_matrix_dimension(h2_molecule: Molecule) -> None:
    """Verify H2 with STO-3G has 2 basis functions.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    assert S.n_basis == 2


def test_h2_off_diagonal_positive(h2_molecule: Molecule) -> None:
    """Verify off-diagonal elements are positive for bonded atoms.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    # S[0,1] is overlap between 1s on H1 and 1s on H2
    assert S[0, 1] > 0
    assert S[0, 1] < 1.0  # Less than diagonal


# ======================================================================
# Water Specific Tests
# ======================================================================


def test_water_overlap_matrix_dimension(water_molecule: Molecule) -> None:
    """Verify water with STO-3G has 7 basis functions.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
    # H1: 1s = 1 function
    # H2: 1s = 1 function
    # Total: 7
    assert S.n_basis == 7


def test_water_overlap_matrix_shape(water_molecule: Molecule) -> None:
    """Verify overlap matrix has correct shape.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    S: Overlap = Overlap(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    assert S.matrix.shape == (7, 7)
