"""Unit tests for q_block.theory.integrals.kinetic_energy module.

Tests cover:
- Primitive kinetic energy integrals for s, p, d orbitals
- Contracted kinetic energy integrals
- Full kinetic energy matrix for molecules
- Symmetry of the matrix
- Positive semi-definiteness (all eigenvalues >= 0)

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block import ContractedGaussianTypeOrbital, Molecule, KineticEnergy, Overlap
from tests.constants import ORIGIN
from tests.theory.integrals.integrals_testing_utils import (
    ALL_ORBITAL_COMPONENTS,
    get_orbital_ids,
    h2_molecule,
    water_molecule,
)


# ======================================================================
# Primitive Kinetic Energy Tests - Parametrized by Orbital Type
# ======================================================================

# ----------------------------------------------------------------------
# Self Kinetic Energy Tests (positive for identical orbitals)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_kinetic_self_positive(lx: int, ly: int, lz: int) -> None:
    """Verify kinetic energy of identical normalized orbital with itself is positive.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0
    kinetic: float = KineticEnergy.primitive_kinetic(
        alpha=alpha, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=alpha, B=A, lx2=lx, ly2=ly, lz2=lz
    )
    assert kinetic > 0


# ----------------------------------------------------------------------
# Same Center, Different Exponents Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_kinetic_same_center_diff_exponents(lx: int, ly: int, lz: int) -> None:
    """Verify two orbitals at same center with different exponents have positive kinetic energy.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    kinetic: float = KineticEnergy.primitive_kinetic(
        alpha=1.0, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=2.0, B=A, lx2=lx, ly2=ly, lz2=lz
    )
    assert kinetic > 0


# ----------------------------------------------------------------------
# Far Apart Orbitals Tests (near-zero kinetic coupling)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_kinetic_far_apart(lx: int, ly: int, lz: int) -> None:
    """Verify orbitals far apart have near-zero kinetic energy coupling.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (0.0, 0.0, 100.0)
    kinetic: float = KineticEnergy.primitive_kinetic(
        alpha=1.0, A=A, lx1=lx, ly1=ly, lz1=lz,
        beta=1.0, B=B, lx2=lx, ly2=ly, lz2=lz
    )
    assert kinetic == pytest.approx(expected=0.0, abs=1e-10)


# ----------------------------------------------------------------------
# Higher Exponent = Higher Kinetic Energy
# ----------------------------------------------------------------------


def test_primitive_kinetic_increases_with_exponent() -> None:
    """Verify kinetic energy increases with exponent for s-orbitals.

    Tighter (higher exponent) Gaussians have higher kinetic energy due to
    faster spatial variation.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Lower exponent (more diffuse)
    T_low: float = KineticEnergy.primitive_kinetic(
        alpha=0.5, A=A, lx1=0, ly1=0, lz1=0,
        beta=0.5, B=A, lx2=0, ly2=0, lz2=0
    )

    # Higher exponent (more compact)
    T_high: float = KineticEnergy.primitive_kinetic(
        alpha=2.0, A=A, lx1=0, ly1=0, lz1=0,
        beta=2.0, B=A, lx2=0, ly2=0, lz2=0
    )

    assert T_low < T_high


# ----------------------------------------------------------------------
# Symmetry Test (T_ab = T_ba)
# ----------------------------------------------------------------------


def test_primitive_kinetic_symmetric() -> None:
    """Verify kinetic energy integral is symmetric: T(a|b) = T(b|a)."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (1.0, 0.5, 0.2)

    T_ab: float = KineticEnergy.primitive_kinetic(
        alpha=1.5, A=A, lx1=1, ly1=0, lz1=0,
        beta=2.0, B=B, lx2=0, ly2=1, lz2=0
    )

    T_ba: float = KineticEnergy.primitive_kinetic(
        alpha=2.0, A=B, lx1=0, ly1=1, lz1=0,
        beta=1.5, B=A, lx2=1, ly2=0, lz2=0
    )

    assert T_ab == pytest.approx(expected=T_ba, rel=1e-10)


# ======================================================================
# Contracted Kinetic Energy Tests
# ======================================================================


def test_contracted_kinetic_single_primitive_equals_primitive() -> None:
    """Verify single-primitive CGTO kinetic matches primitive kinetic."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    kinetic_contracted: float = KineticEnergy.contracted_kinetic(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0
    )
    kinetic_primitive: float = KineticEnergy.primitive_kinetic(
        alpha=1.0, A=(0.0, 0.0, 0.0), lx1=0, ly1=0, lz1=0,
        beta=1.0, B=(0.0, 0.0, 0.0), lx2=0, ly2=0, lz2=0
    )
    assert kinetic_contracted == pytest.approx(expected=kinetic_primitive, rel=1e-10)


def test_contracted_kinetic_multiple_primitives_positive() -> None:
    """Verify multiple contracted primitives give positive self-kinetic energy."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[3.0, 1.0, 0.3],
        contractions=[0.5, 0.3, 0.2],
    )
    kinetic: float = KineticEnergy.contracted_kinetic(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0
    )
    assert kinetic > 0


# ======================================================================
# Matrix Initialization Tests
# ======================================================================


def test_kinetic_matrix_empty_basis_raises() -> None:
    """Verify empty basis set raises ValueError."""
    with pytest.raises(expected_exception=ValueError, match="empty basis set"):
        KineticEnergy(cgtos=[])


def test_kinetic_matrix_single_s_orbital() -> None:
    """Verify single s-orbital gives 1x1 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    T: KineticEnergy = KineticEnergy(cgtos=[cgto])
    assert T.n_basis == 1
    assert T.matrix.shape == (1, 1)


def test_kinetic_matrix_single_p_orbital() -> None:
    """Verify single p-shell gives 3x3 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=1,
        exponents=[1.0],
        contractions=[1.0],
    )
    T: KineticEnergy = KineticEnergy(cgtos=[cgto])
    assert T.n_basis == 3
    assert T.matrix.shape == (3, 3)


# ======================================================================
# Matrix Symmetry Tests
# ======================================================================


def test_kinetic_matrix_symmetry_h2(h2_molecule: Molecule) -> None:
    """Verify kinetic matrix is symmetric for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(actual=T.matrix, desired=T.matrix.T)


def test_kinetic_matrix_symmetry_water(water_molecule: Molecule) -> None:
    """Verify kinetic matrix is symmetric for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    np.testing.assert_array_almost_equal(actual=T.matrix, desired=T.matrix.T)


# ======================================================================
# Matrix Diagonal Tests (Positive Kinetic Energy)
# ======================================================================


def test_kinetic_matrix_diagonal_positive_h2(h2_molecule: Molecule) -> None:
    """Verify diagonal elements are positive for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(v=T.matrix)
    assert all(d > 0 for d in diagonal)


def test_kinetic_matrix_diagonal_positive_water(water_molecule: Molecule) -> None:
    """Verify diagonal elements are positive for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    diagonal: np.ndarray = np.diag(v=T.matrix)
    assert all(d > 0 for d in diagonal)


# ======================================================================
# Matrix Properties Tests
# ======================================================================


def test_kinetic_matrix_positive_semidefinite_h2(h2_molecule: Molecule) -> None:
    """Verify kinetic matrix is positive semi-definite.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    eigenvalues: np.ndarray = np.linalg.eigvalsh(a=T.matrix)
    assert all(ev >= -1e-10 for ev in eigenvalues)


def test_kinetic_matrix_positive_semidefinite_water(water_molecule: Molecule) -> None:
    """Verify kinetic matrix is positive semi-definite.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    eigenvalues: np.ndarray = np.linalg.eigvalsh(a=T.matrix)
    assert all(ev >= -1e-10 for ev in eigenvalues)


def test_kinetic_matrix_repr(h2_molecule: Molecule) -> None:
    """Verify __repr__ output.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    repr_str: str = repr(T)
    assert "KineticEnergy" in repr_str
    assert "n_basis" in repr_str


def test_kinetic_matrix_getitem(h2_molecule: Molecule) -> None:
    """Verify __getitem__ access.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    assert T[0, 0] == T.matrix[0, 0]
    assert T[0, 1] == T.matrix[0, 1]


# ======================================================================
# H2 Specific Tests
# ======================================================================


def test_h2_kinetic_matrix_dimension(h2_molecule: Molecule) -> None:
    """Verify H2 with STO-3G has 2 basis functions.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    assert T.n_basis == 2


def test_h2_kinetic_off_diagonal_positive(h2_molecule: Molecule) -> None:
    """Verify off-diagonal elements are positive for bonded atoms.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    # T[0,1] is kinetic coupling between 1s on H1 and 1s on H2
    assert T[0, 1] > 0


# ======================================================================
# Water Specific Tests
# ======================================================================


def test_water_kinetic_matrix_dimension(water_molecule: Molecule) -> None:
    """Verify water with STO-3G has 7 basis functions.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
    # H1: 1s = 1 function
    # H2: 1s = 1 function
    # Total: 7
    assert T.n_basis == 7


def test_water_kinetic_matrix_shape(water_molecule: Molecule) -> None:
    """Verify kinetic matrix has correct shape.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=water_molecule.contracted_gaussian_type_orbitals)
    assert T.matrix.shape == (7, 7)


# ======================================================================
# Comparison Tests (Kinetic vs Overlap)
# ======================================================================


def test_kinetic_vs_overlap_different_magnitudes(h2_molecule: Molecule) -> None:
    """Verify T and S matrices have different magnitudes.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    T: KineticEnergy = KineticEnergy(cgtos=h2_molecule.contracted_gaussian_type_orbitals)
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)

    # Diagonal of S should be ~1, diagonal of T should be related to kinetic energy
    assert not np.allclose(T.matrix, S.matrix)
    # But same shape
    assert T.matrix.shape == S.matrix.shape
