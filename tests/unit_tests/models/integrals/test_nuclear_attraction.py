"""Unit tests for q_block.models.integrals.nuclear_attraction module.

Tests cover:
- Primitive nuclear attraction integrals for s, p, d orbitals
- Contracted nuclear attraction integrals
- Full nuclear attraction matrix for molecules
- Symmetry of the matrix
- Negative semi-definiteness (all eigenvalues <= 0)
- Negative diagonal elements (attraction is always negative)

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block import (
    ContractedGaussianTypeOrbital,
    Molecule,
    NuclearAttraction,
    Overlap,
)
from tests.unit_tests.environment.constants import ORIGIN
from tests.unit_tests.utilities import (
    ALL_ORBITAL_COMPONENTS,
    get_orbital_ids,
    h2_molecule,
    h2_nuclei,
    water_molecule,
    water_nuclei,
)

# ======================================================================
# Primitive Nuclear Attraction Tests - Parametrized by Orbital Type
# ======================================================================

# ----------------------------------------------------------------------
# Self Nuclear Attraction Tests (negative for identical orbitals)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_nuclear_attraction_self_negative(
    lx: int, ly: int, lz: int
) -> None:
    """Verify nuclear attraction of identical orbital with itself is negative.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    C: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Nucleus at same point
    alpha: float = 1.0
    Z: int = 1

    # Nucleus slightly offset to avoid numerical issues at r=0
    C = (0.01, 0.0, 0.0)

    attraction: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=alpha,
        A=A,
        lx1=lx,
        ly1=ly,
        lz1=lz,
        beta=alpha,
        B=A,
        lx2=lx,
        ly2=ly,
        lz2=lz,
        C=C,
        Z=Z,
    )
    assert attraction < 0


# ----------------------------------------------------------------------
# Far Apart Orbitals Tests (near-zero coupling)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "lx, ly, lz",
    ALL_ORBITAL_COMPONENTS,
    ids=get_orbital_ids(),
)
def test_primitive_nuclear_attraction_far_apart(lx: int, ly: int, lz: int) -> None:
    """Verify orbitals far apart have near-zero nuclear attraction coupling.

    :param lx: int - Angular momentum component in x direction.
    :param ly: int - Angular momentum component in y direction.
    :param lz: int - Angular momentum component in z direction.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (0.0, 0.0, 100.0)
    C: tuple[float, float, float] = (0.0, 0.0, 50.0)  # Nucleus between orbitals

    attraction: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=1.0,
        A=A,
        lx1=lx,
        ly1=ly,
        lz1=lz,
        beta=1.0,
        B=B,
        lx2=lx,
        ly2=ly,
        lz2=lz,
        C=C,
        Z=1,
    )
    assert attraction == pytest.approx(expected=0.0, abs=1e-8)


# ----------------------------------------------------------------------
# Nuclear Charge Scaling Test
# ----------------------------------------------------------------------


def test_primitive_nuclear_attraction_scales_with_Z() -> None:
    """Verify nuclear attraction scales linearly with nuclear charge Z."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    C: tuple[float, float, float] = (0.5, 0.0, 0.0)

    V_Z1: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=1.0,
        A=A,
        lx1=0,
        ly1=0,
        lz1=0,
        beta=1.0,
        B=A,
        lx2=0,
        ly2=0,
        lz2=0,
        C=C,
        Z=1,
    )

    V_Z2: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=1.0,
        A=A,
        lx1=0,
        ly1=0,
        lz1=0,
        beta=1.0,
        B=A,
        lx2=0,
        ly2=0,
        lz2=0,
        C=C,
        Z=2,
    )

    assert V_Z2 == pytest.approx(expected=2 * V_Z1, rel=1e-10)


# ----------------------------------------------------------------------
# Symmetry Test (V_ab = V_ba)
# ----------------------------------------------------------------------


def test_primitive_nuclear_attraction_symmetric() -> None:
    """Verify nuclear attraction integral is symmetric: V(a|b) = V(b|a)."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (1.0, 0.5, 0.2)
    C: tuple[float, float, float] = (0.3, 0.1, 0.0)

    V_ab: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=1.5,
        A=A,
        lx1=1,
        ly1=0,
        lz1=0,
        beta=2.0,
        B=B,
        lx2=0,
        ly2=1,
        lz2=0,
        C=C,
        Z=1,
    )

    V_ba: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=2.0,
        A=B,
        lx1=0,
        ly1=1,
        lz1=0,
        beta=1.5,
        B=A,
        lx2=1,
        ly2=0,
        lz2=0,
        C=C,
        Z=1,
    )

    assert V_ab == pytest.approx(expected=V_ba, rel=1e-10)


# ======================================================================
# Contracted Nuclear Attraction Tests
# ======================================================================


def test_contracted_nuclear_attraction_single_primitive_equals_primitive() -> None:
    """Verify single-primitive CGTO nuclear attraction matches primitive."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    C: tuple[float, float, float] = (0.5, 0.0, 0.0)
    Z: int = 1

    attraction_contracted: float = NuclearAttraction.contracted_nuclear_attraction(
        cgto1=cgto, lx1=0, ly1=0, lz1=0, cgto2=cgto, lx2=0, ly2=0, lz2=0, C=C, Z=Z
    )
    attraction_primitive: float = NuclearAttraction.primitive_nuclear_attraction(
        alpha=1.0,
        A=(0.0, 0.0, 0.0),
        lx1=0,
        ly1=0,
        lz1=0,
        beta=1.0,
        B=(0.0, 0.0, 0.0),
        lx2=0,
        ly2=0,
        lz2=0,
        C=C,
        Z=Z,
    )
    assert attraction_contracted == pytest.approx(
        expected=attraction_primitive, rel=1e-10
    )


def test_contracted_nuclear_attraction_negative() -> None:
    """Verify contracted self-attraction is negative."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[3.0, 1.0, 0.3],
        contractions=[0.5, 0.3, 0.2],
    )
    C: tuple[float, float, float] = (0.1, 0.0, 0.0)

    attraction: float = NuclearAttraction.contracted_nuclear_attraction(
        cgto1=cgto, lx1=0, ly1=0, lz1=0, cgto2=cgto, lx2=0, ly2=0, lz2=0, C=C, Z=1
    )
    assert attraction < 0


# ======================================================================
# Matrix Initialization Tests
# ======================================================================


def test_nuclear_attraction_matrix_empty_basis_raises() -> None:
    """Verify empty basis set raises ValueError."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    with pytest.raises(expected_exception=ValueError, match="empty basis set"):
        NuclearAttraction(cgtos=[], nuclei=nuclei)


def test_nuclear_attraction_matrix_empty_nuclei_raises() -> None:
    """Verify empty nuclei list raises ValueError."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    with pytest.raises(expected_exception=ValueError, match="without nuclei"):
        NuclearAttraction(cgtos=[cgto], nuclei=[])


def test_nuclear_attraction_matrix_single_s_orbital() -> None:
    """Verify single s-orbital gives 1x1 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    nuclei = [(1, (0.5, 0.0, 0.0))]
    V: NuclearAttraction = NuclearAttraction(cgtos=[cgto], nuclei=nuclei)
    assert V.n_basis == 1
    assert V.matrix.shape == (1, 1)


def test_nuclear_attraction_matrix_single_p_orbital() -> None:
    """Verify single p-shell gives 3x3 matrix."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=1,
        exponents=[1.0],
        contractions=[1.0],
    )
    nuclei = [(1, (0.5, 0.0, 0.0))]
    V: NuclearAttraction = NuclearAttraction(cgtos=[cgto], nuclei=nuclei)
    assert V.n_basis == 3
    assert V.matrix.shape == (3, 3)


# ======================================================================
# Matrix Symmetry Tests
# ======================================================================


def test_nuclear_attraction_matrix_symmetry_h2(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify nuclear attraction matrix is symmetric for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    np.testing.assert_array_almost_equal(actual=V.matrix, desired=V.matrix.T)


def test_nuclear_attraction_matrix_symmetry_water(
    water_molecule: Molecule, water_nuclei: list
) -> None:
    """Verify nuclear attraction matrix is symmetric for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    :param water_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=water_molecule.contracted_gaussian_type_orbitals, nuclei=water_nuclei
    )
    np.testing.assert_array_almost_equal(actual=V.matrix, desired=V.matrix.T)


# ======================================================================
# Matrix Diagonal Tests (Negative Attraction)
# ======================================================================


def test_nuclear_attraction_matrix_diagonal_negative_h2(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify diagonal elements are negative for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    diagonal: np.ndarray = np.diag(v=V.matrix)
    assert all(d < 0 for d in diagonal)


def test_nuclear_attraction_matrix_diagonal_negative_water(
    water_molecule: Molecule, water_nuclei: list
) -> None:
    """Verify diagonal elements are negative for water.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    :param water_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=water_molecule.contracted_gaussian_type_orbitals, nuclei=water_nuclei
    )
    diagonal: np.ndarray = np.diag(v=V.matrix)
    assert all(d < 0 for d in diagonal)


# ======================================================================
# Matrix Properties Tests
# ======================================================================


def test_nuclear_attraction_matrix_negative_semidefinite_h2(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify nuclear attraction matrix is negative semi-definite.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    eigenvalues: np.ndarray = np.linalg.eigvalsh(a=V.matrix)
    assert all(ev <= 1e-10 for ev in eigenvalues)


def test_nuclear_attraction_matrix_negative_semidefinite_water(
    water_molecule: Molecule, water_nuclei: list
) -> None:
    """Verify nuclear attraction matrix is negative semi-definite.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    :param water_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=water_molecule.contracted_gaussian_type_orbitals, nuclei=water_nuclei
    )
    eigenvalues: np.ndarray = np.linalg.eigvalsh(a=V.matrix)
    assert all(ev <= 1e-10 for ev in eigenvalues)


def test_nuclear_attraction_matrix_repr(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify __repr__ output.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    repr_str: str = repr(V)
    assert "NuclearAttraction" in repr_str
    assert "n_basis" in repr_str


def test_nuclear_attraction_matrix_getitem(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify __getitem__ access.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    assert V[0, 0] == V.matrix[0, 0]
    assert V[0, 1] == V.matrix[0, 1]


# ======================================================================
# H2 Specific Tests
# ======================================================================


def test_h2_nuclear_attraction_matrix_dimension(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify H2 with STO-3G has 2 basis functions.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals, nuclei=h2_nuclei
    )
    assert V.n_basis == 2


# ======================================================================
# Water Specific Tests
# ======================================================================


def test_water_nuclear_attraction_matrix_dimension(
    water_molecule: Molecule, water_nuclei: list
) -> None:
    """Verify water with STO-3G has 7 basis functions.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    :param water_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=water_molecule.contracted_gaussian_type_orbitals, nuclei=water_nuclei
    )
    # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
    # H1: 1s = 1 function
    # H2: 1s = 1 function
    # Total: 7
    assert V.n_basis == 7


def test_water_nuclear_attraction_matrix_shape(
    water_molecule: Molecule, water_nuclei: list
) -> None:
    """Verify nuclear attraction matrix has correct shape.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    :param water_nuclei: list - Nuclear positions and charges.
    """
    V: NuclearAttraction = NuclearAttraction(
        cgtos=water_molecule.contracted_gaussian_type_orbitals, nuclei=water_nuclei
    )
    assert V.matrix.shape == (7, 7)


# ======================================================================
# Nuclear Count Tests
# ======================================================================


def test_nuclear_attraction_multiple_nuclei_additive(
    h2_molecule: Molecule, h2_nuclei: list
) -> None:
    """Verify nuclear attraction is sum over individual nuclei.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    :param h2_nuclei: list - Nuclear positions and charges.
    """
    cgtos = h2_molecule.contracted_gaussian_type_orbitals

    # Full V matrix (sum over both nuclei)
    V_full: NuclearAttraction = NuclearAttraction(cgtos=cgtos, nuclei=h2_nuclei)

    # Individual V matrices
    V_H1: NuclearAttraction = NuclearAttraction(cgtos=cgtos, nuclei=[h2_nuclei[0]])
    V_H2: NuclearAttraction = NuclearAttraction(cgtos=cgtos, nuclei=[h2_nuclei[1]])

    # V_full should equal V_H1 + V_H2
    np.testing.assert_array_almost_equal(
        actual=V_full.matrix, desired=V_H1.matrix + V_H2.matrix, decimal=10
    )
