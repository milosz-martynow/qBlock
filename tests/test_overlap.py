"""Unit tests for OverlapMatrix and overlap integral computation.

Tests cover:
- Primitive overlap integrals for s, p, d orbitals
- Contracted overlap integrals
- Full overlap matrix for molecules
- Normalization (diagonal elements should be ~1.0)
- Symmetry of the overlap matrix
"""

import math

import pytest
import numpy as np

from q_block import Molecule, OverlapMatrix, ContractedGaussianTypeOrbital
from q_block.io.basis_set import Pople
from q_block.io.coordinates import CartesianCoordinates
from q_block.io.input_data import InputData
from q_block.theory.utils import get_cartesian_components
from scipy.special import factorial2


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def basis_sto3g() -> Pople:
    """Load the STO-3G basis set."""
    return Pople(filepath="data/basis_set/sto_gaussian_format/STO-3G.gbs")


@pytest.fixture
def basis_631g() -> Pople:
    """Load the 6-31G basis set."""
    return Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")


@pytest.fixture
def origin() -> CartesianCoordinates:
    """Origin coordinate."""
    return CartesianCoordinates(0.0, 0.0, 0.0)


@pytest.fixture
def h2_molecule(basis_sto3g: Pople) -> Molecule:
    """H2 molecule with STO-3G basis, converted to Bohr."""
    inp = InputData()
    inp.from_script(atom_data=[
        ["H", 0.0, 0.0, 0.0, basis_sto3g],
        ["H", 0.0, 0.0, 0.74, basis_sto3g],  # ~0.74 Å bond length
    ])
    mol = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol


@pytest.fixture
def water_molecule(basis_sto3g: Pople) -> Molecule:
    """Water molecule with STO-3G basis, converted to Bohr."""
    inp = InputData()
    inp.from_script(atom_data=[
        ["O", 0.0, 0.0, 0.1173, basis_sto3g],
        ["H", 0.0, 0.7572, -0.4692, basis_sto3g],
        ["H", 0.0, -0.7572, -0.4692, basis_sto3g],
    ])
    mol = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol

class TestGaussianProductCenter:
    """Tests for Gaussian product center calculation."""

    def test_same_center(self):
        """Product center of identical positions is that position."""
        A = (1.0, 2.0, 3.0)
        P = OverlapMatrix._gaussian_product_center(1.0, A, 1.0, A)
        assert P == pytest.approx(A)

    def test_equal_exponents(self):
        """Equal exponents give midpoint."""
        A = (0.0, 0.0, 0.0)
        B = (2.0, 0.0, 0.0)
        P = OverlapMatrix._gaussian_product_center(1.0, A, 1.0, B)
        assert P == pytest.approx((1.0, 0.0, 0.0))

    def test_unequal_exponents(self):
        """Larger exponent pulls center toward that Gaussian."""
        A = (0.0, 0.0, 0.0)
        B = (2.0, 0.0, 0.0)
        P = OverlapMatrix._gaussian_product_center(3.0, A, 1.0, B)  # alpha=3, beta=1
        # P = (3*0 + 1*2) / (3+1) = 0.5
        assert P == pytest.approx((0.5, 0.0, 0.0))


class TestCartesianComponents:
    """Tests for get_cartesian_components function."""

    def test_s_orbital(self):
        """s-orbital (l=0) has one component."""
        components = get_cartesian_components(0)
        assert components == [(0, 0, 0)]

    def test_p_orbital(self):
        """p-orbital (l=1) has three components."""
        components = get_cartesian_components(1)
        assert len(components) == 3
        assert (1, 0, 0) in components
        assert (0, 1, 0) in components
        assert (0, 0, 1) in components

    def test_d_orbital(self):
        """d-orbital (l=2) has six Cartesian components."""
        components = get_cartesian_components(2)
        assert len(components) == 6
        # Check some expected components
        assert (2, 0, 0) in components  # d_x2
        assert (0, 2, 0) in components  # d_y2
        assert (0, 0, 2) in components  # d_z2
        assert (1, 1, 0) in components  # d_xy
        assert (1, 0, 1) in components  # d_xz
        assert (0, 1, 1) in components  # d_yz


# ======================================================================
# Primitive overlap tests
# ======================================================================


class TestPrimitiveOverlap:
    """Tests for primitive Gaussian overlap integrals."""

    def test_identical_s_orbitals(self):
        """Overlap of identical normalized s-orbital with itself is 1.0."""
        # Single normalized s-type Gaussian at origin
        A = (0.0, 0.0, 0.0)
        alpha = 1.0
        overlap = OverlapMatrix.primitive_overlap(alpha, A, 0, 0, 0, alpha, A, 0, 0, 0)
        assert overlap == pytest.approx(1.0, rel=1e-6)

    def test_s_orbitals_same_center_different_exponents(self):
        """Two s-orbitals at same center have positive overlap."""
        A = (0.0, 0.0, 0.0)
        overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 2.0, A, 0, 0, 0)
        assert overlap > 0
        assert overlap < 1.0  # Different exponents don't give 1.0

    def test_s_orbitals_separated(self):
        """s-orbitals separated along z-axis have reduced overlap."""
        A = (0.0, 0.0, 0.0)
        B = (0.0, 0.0, 2.0)
        overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 1.0, B, 0, 0, 0)
        assert overlap > 0
        assert overlap < 1.0

    def test_s_orbitals_far_apart(self):
        """s-orbitals far apart have near-zero overlap."""
        A = (0.0, 0.0, 0.0)
        B = (0.0, 0.0, 100.0)
        overlap = OverlapMatrix.primitive_overlap(1.0, A, 0, 0, 0, 1.0, B, 0, 0, 0)
        assert overlap == pytest.approx(0.0, abs=1e-10)

    def test_p_orbital_self_overlap(self):
        """p_x orbital overlaps with itself (normalized)."""
        A = (0.0, 0.0, 0.0)
        alpha = 1.0
        # px: lx=1, ly=0, lz=0
        overlap = OverlapMatrix.primitive_overlap(alpha, A, 1, 0, 0, alpha, A, 1, 0, 0)
        assert overlap == pytest.approx(1.0, rel=1e-6)

    def test_orthogonal_p_orbitals(self):
        """Orthogonal p-orbitals (px and py) on same center have zero overlap."""
        A = (0.0, 0.0, 0.0)
        alpha = 1.0
        # px vs py should be zero
        overlap = OverlapMatrix.primitive_overlap(alpha, A, 1, 0, 0, alpha, A, 0, 1, 0)
        assert overlap == pytest.approx(0.0, abs=1e-10)

    def test_s_and_p_same_center(self):
        """s and p orbitals on same center are orthogonal."""
        A = (0.0, 0.0, 0.0)
        alpha = 1.0
        overlap = OverlapMatrix.primitive_overlap(alpha, A, 0, 0, 0, alpha, A, 1, 0, 0)
        assert overlap == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# Contracted overlap tests
# ======================================================================


class TestContractedOverlap:
    """Tests for contracted Gaussian overlap integrals."""

    def test_single_primitive_equals_primitive(self, origin: CartesianCoordinates):
        """Single-primitive CGTO overlap matches primitive overlap."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0],
            contractions=[1.0],
        )
        overlap = OverlapMatrix.contracted_overlap(cgto, 0, 0, 0, cgto, 0, 0, 0)
        primitive = OverlapMatrix.primitive_overlap(
            1.0, (0.0, 0.0, 0.0), 0, 0, 0,
            1.0, (0.0, 0.0, 0.0), 0, 0, 0
        )
        assert overlap == pytest.approx(primitive, rel=1e-10)

    def test_multiple_primitives_positive(self, origin: CartesianCoordinates):
        """Multiple contracted primitives still give positive self-overlap."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[3.0, 1.0, 0.3],
            contractions=[0.5, 0.3, 0.2],
        )
        overlap = OverlapMatrix.contracted_overlap(cgto, 0, 0, 0, cgto, 0, 0, 0)
        assert overlap > 0


# ======================================================================
# OverlapMatrix class tests
# ======================================================================


class TestOverlapMatrixInit:
    """Tests for OverlapMatrix initialization."""

    def test_empty_basis_raises(self):
        """Empty basis set raises ValueError."""
        with pytest.raises(ValueError, match="empty basis set"):
            OverlapMatrix([])

    def test_single_s_orbital(self, origin: CartesianCoordinates):
        """Single s-orbital gives 1x1 matrix."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0],
            contractions=[1.0],
        )
        S = OverlapMatrix([cgto])
        assert S.n_basis == 1
        assert S.matrix.shape == (1, 1)

    def test_single_p_orbital(self, origin: CartesianCoordinates):
        """Single p-shell gives 3x3 matrix."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=1,
            exponents=[1.0],
            contractions=[1.0],
        )
        S = OverlapMatrix([cgto])
        assert S.n_basis == 3
        assert S.matrix.shape == (3, 3)


class TestOverlapMatrixSymmetry:
    """Tests for overlap matrix symmetry."""

    def test_symmetry_h2(self, h2_molecule: Molecule):
        """Overlap matrix is symmetric for H2."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        np.testing.assert_array_almost_equal(S.matrix, S.matrix.T)

    def test_symmetry_water(self, water_molecule: Molecule):
        """Overlap matrix is symmetric for water."""
        S = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
        np.testing.assert_array_almost_equal(S.matrix, S.matrix.T)


class TestOverlapMatrixDiagonal:
    """Tests for overlap matrix diagonal elements (normalization)."""

    def test_diagonal_ones_h2(self, h2_molecule: Molecule):
        """Diagonal elements should be close to 1.0 for normalized basis."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        diagonal = np.diag(S.matrix)
        np.testing.assert_array_almost_equal(diagonal, np.ones(S.n_basis), decimal=4)

    def test_diagonal_ones_water(self, water_molecule: Molecule):
        """Diagonal elements should be close to 1.0 for water."""
        S = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
        diagonal = np.diag(S.matrix)
        np.testing.assert_array_almost_equal(diagonal, np.ones(S.n_basis), decimal=4)


class TestOverlapMatrixProperties:
    """Tests for general overlap matrix properties."""

    def test_positive_definite_h2(self, h2_molecule: Molecule):
        """Overlap matrix should be positive definite."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        eigenvalues = np.linalg.eigvalsh(S.matrix)
        assert all(ev > 0 for ev in eigenvalues)

    def test_repr(self, h2_molecule: Molecule):
        """Test __repr__ output."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        repr_str = repr(S)
        assert "OverlapMatrix" in repr_str
        assert "n_basis" in repr_str

    def test_getitem(self, h2_molecule: Molecule):
        """Test __getitem__ access."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        assert S[0, 0] == S.matrix[0, 0]
        assert S[0, 1] == S.matrix[0, 1]


class TestOverlapMatrixH2:
    """Specific tests for H2 overlap matrix with STO-3G."""

    def test_h2_dimension(self, h2_molecule: Molecule):
        """H2 with STO-3G has 2 basis functions (one per H)."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        assert S.n_basis == 2

    def test_h2_off_diagonal_positive(self, h2_molecule: Molecule):
        """Off-diagonal elements should be positive for bonded atoms."""
        S = OverlapMatrix(h2_molecule.contracted_gaussian_type_orbitals)
        # S[0,1] is overlap between 1s on H1 and 1s on H2
        assert S[0, 1] > 0
        assert S[0, 1] < 1.0  # Less than diagonal


class TestOverlapMatrixWater:
    """Specific tests for water overlap matrix with STO-3G."""

    def test_water_dimension(self, water_molecule: Molecule):
        """Water with STO-3G has 7 basis functions (5 on O, 1 per H)."""
        S = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
        # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
        # H1: 1s = 1 function
        # H2: 1s = 1 function
        # Total: 7
        assert S.n_basis == 7

    def test_water_shape(self, water_molecule: Molecule):
        """Overlap matrix has correct shape."""
        S = OverlapMatrix(water_molecule.contracted_gaussian_type_orbitals)
        assert S.matrix.shape == (7, 7)
