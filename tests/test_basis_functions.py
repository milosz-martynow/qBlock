"""Unit tests for ContractedGaussianTypeOrbital and related CGTO functionality.

Tests cover:
- ContractedGaussianTypeOrbital class initialization, validation, and methods
- Atom.make_contracted_gaussian_type_orbital
- Molecule.make_contracted_gaussian_type_orbital
- Molecule.molecular_orbital
"""

import math
from typing import List, Tuple

import pytest

from q_block import Atom, Molecule, ContractedGaussianTypeOrbital
from q_block.io.basis_set import Pople
from q_block.io.coordinates import CartesianCoordinates
from q_block.io.input_data import InputData


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def basis_631g() -> Pople:
    """Load the 6-31G basis set."""
    return Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")


@pytest.fixture
def basis_sto3g() -> Pople:
    """Load the STO-3G basis set."""
    return Pople(filepath="data/basis_set/sto_gaussian_format/STO-3G.gbs")


@pytest.fixture
def origin() -> CartesianCoordinates:
    """Origin coordinate."""
    return CartesianCoordinates(0.0, 0.0, 0.0)


@pytest.fixture
def simple_cgto(origin: CartesianCoordinates) -> ContractedGaussianTypeOrbital:
    """A simple s-type CGTO with one primitive for testing."""
    return ContractedGaussianTypeOrbital(
        center=origin,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
        atom_index=0,
    )


@pytest.fixture
def hydrogen_atom(basis_631g: Pople, origin: CartesianCoordinates) -> Atom:
    """Hydrogen atom at origin with 6-31G basis."""
    return Atom(
        atomic_number=1,
        basis_set=basis_631g,
        coordinates=origin,
    )


@pytest.fixture
def water_molecule(basis_631g: Pople) -> Molecule:
    """Water molecule with 6-31G basis."""
    inp = InputData()
    inp.from_script(atom_data=[
        ["O", 0.0, 0.0, 0.1173, basis_631g],
        ["H", 0.0, 0.7572, -0.4692, basis_631g],
        ["H", 0.0, -0.7572, -0.4692, basis_631g],
    ])
    return Molecule(input_data=inp)


# ======================================================================
# ContractedGaussianTypeOrbital - Initialization Tests
# ======================================================================


class TestCGTOInit:
    """Tests for ContractedGaussianTypeOrbital initialization."""

    def test_init_basic(self, origin: CartesianCoordinates):
        """Test basic CGTO initialization."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0, 0.5],
            contractions=[0.6, 0.4],
            atom_index=0,
        )
        assert cgto.center == origin
        assert cgto.l == 0
        assert cgto.exponents == [1.0, 0.5]
        assert cgto.contractions == [0.6, 0.4]
        assert cgto.atom_index == 0
        assert cgto.n_primitives == 2
        assert cgto.n_functions == 1  # 2*0 + 1 = 1

    def test_init_p_orbital(self, origin: CartesianCoordinates):
        """Test CGTO initialization for p-orbital (l=1)."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=1,
            exponents=[1.0],
            contractions=[1.0],
        )
        assert cgto.l == 1
        assert cgto.n_functions == 3  # 2*1 + 1 = 3

    def test_init_d_orbital(self, origin: CartesianCoordinates):
        """Test CGTO initialization for d-orbital (l=2)."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=2,
            exponents=[1.0],
            contractions=[1.0],
        )
        assert cgto.l == 2
        assert cgto.n_functions == 5  # 2*2 + 1 = 5

    def test_init_atom_index_none(self, origin: CartesianCoordinates):
        """Test that atom_index defaults to None."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0],
            contractions=[1.0],
        )
        assert cgto.atom_index is None

    def test_init_length_mismatch_raises(self, origin: CartesianCoordinates):
        """Test that mismatched exponents/contractions raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            ContractedGaussianTypeOrbital(
                center=origin,
                l=0,
                exponents=[1.0, 0.5],
                contractions=[1.0],
            )

    def test_init_empty_exponents_raises(self, origin: CartesianCoordinates):
        """Test that empty exponents raises ValueError."""
        with pytest.raises(ValueError, match="at least one primitive"):
            ContractedGaussianTypeOrbital(
                center=origin,
                l=0,
                exponents=[],
                contractions=[],
            )


# ======================================================================
# ContractedGaussianTypeOrbital - Dunder Methods Tests
# ======================================================================


class TestCGTODunder:
    """Tests for CGTO dunder methods (__repr__, __eq__)."""

    def test_repr_s_orbital(self, simple_cgto: ContractedGaussianTypeOrbital):
        """Test __repr__ for s-orbital."""
        repr_str = repr(simple_cgto)
        assert "CGTO" in repr_str
        assert "atom=0" in repr_str
        assert "s" in repr_str
        assert "K=1" in repr_str

    def test_repr_p_orbital(self, origin: CartesianCoordinates):
        """Test __repr__ for p-orbital."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=1,
            exponents=[1.0],
            contractions=[1.0],
        )
        assert "p" in repr(cgto)

    def test_repr_d_orbital(self, origin: CartesianCoordinates):
        """Test __repr__ for d-orbital."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=2,
            exponents=[1.0],
            contractions=[1.0],
        )
        assert "d" in repr(cgto)

    def test_eq_same(self, origin: CartesianCoordinates):
        """Test equality of identical CGTOs."""
        cgto1 = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0],
            contractions=[1.0],
            atom_index=0,
        )
        cgto2 = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0],
            contractions=[1.0],
            atom_index=0,
        )
        assert cgto1 == cgto2

    def test_eq_different_l(self, origin: CartesianCoordinates):
        """Test inequality when l differs."""
        cgto1 = ContractedGaussianTypeOrbital(
            center=origin, l=0, exponents=[1.0], contractions=[1.0]
        )
        cgto2 = ContractedGaussianTypeOrbital(
            center=origin, l=1, exponents=[1.0], contractions=[1.0]
        )
        assert cgto1 != cgto2

    def test_eq_different_exponents(self, origin: CartesianCoordinates):
        """Test inequality when exponents differ."""
        cgto1 = ContractedGaussianTypeOrbital(
            center=origin, l=0, exponents=[1.0], contractions=[1.0]
        )
        cgto2 = ContractedGaussianTypeOrbital(
            center=origin, l=0, exponents=[2.0], contractions=[1.0]
        )
        assert cgto1 != cgto2

    def test_eq_not_cgto(self, simple_cgto: ContractedGaussianTypeOrbital):
        """Test comparison with non-CGTO object returns NotImplemented."""
        assert simple_cgto.__eq__("not a cgto") == NotImplemented


# ======================================================================
# ContractedGaussianTypeOrbital - Static Method Tests
# ======================================================================


class TestCGTOStaticMethods:
    """Tests for CGTO static helper methods."""

    def test_double_factorial_negative_one(self):
        """Test double factorial of -1 equals 1."""
        assert ContractedGaussianTypeOrbital._double_factorial(-1) == 1

    def test_double_factorial_zero(self):
        """Test double factorial of 0 equals 1."""
        assert ContractedGaussianTypeOrbital._double_factorial(0) == 1

    def test_double_factorial_one(self):
        """Test double factorial of 1 equals 1."""
        assert ContractedGaussianTypeOrbital._double_factorial(1) == 1

    def test_double_factorial_five(self):
        """Test double factorial of 5 equals 15 (5*3*1)."""
        assert ContractedGaussianTypeOrbital._double_factorial(5) == 15

    def test_double_factorial_six(self):
        """Test double factorial of 6 equals 48 (6*4*2)."""
        assert ContractedGaussianTypeOrbital._double_factorial(6) == 48

    def test_normalisation_constant_s_orbital(self):
        """Test normalisation constant for s-orbital (l=0)."""
        alpha = 1.0
        norm = ContractedGaussianTypeOrbital._normalisation_constant(alpha, 0, 0, 0)
        # For s-orbital: N = (2*alpha/pi)^(3/4)
        expected = (2.0 * alpha / math.pi) ** 0.75
        assert abs(norm - expected) < 1e-10

    def test_normalisation_constant_positive(self):
        """Test normalisation constant is positive."""
        for lx in range(3):
            for ly in range(3 - lx):
                lz = 2 - lx - ly
                if lx + ly + lz <= 2:
                    norm = ContractedGaussianTypeOrbital._normalisation_constant(
                        1.0, lx, ly, lz
                    )
                    assert norm > 0


# ======================================================================
# ContractedGaussianTypeOrbital - Evaluation Tests
# ======================================================================


class TestCGTOEvaluation:
    """Tests for CGTO primitive_gaussian and basis_function methods."""

    def test_primitive_gaussian_at_center(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test primitive Gaussian at the center is the normalisation constant."""
        value = simple_cgto.primitive_gaussian(origin, 0, 0, 0, 0)
        expected = ContractedGaussianTypeOrbital._normalisation_constant(1.0, 0, 0, 0)
        assert abs(value - expected) < 1e-10

    def test_primitive_gaussian_decays(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test primitive Gaussian decays away from center."""
        val_0 = simple_cgto.primitive_gaussian(origin, 0, 0, 0, 0)
        val_1 = simple_cgto.primitive_gaussian(
            CartesianCoordinates(1.0, 0.0, 0.0), 0, 0, 0, 0
        )
        val_2 = simple_cgto.primitive_gaussian(
            CartesianCoordinates(2.0, 0.0, 0.0), 0, 0, 0, 0
        )
        assert val_0 > val_1 > val_2 > 0

    def test_primitive_gaussian_index_error(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test primitive_gaussian raises IndexError for invalid index."""
        with pytest.raises(IndexError, match="out of range"):
            simple_cgto.primitive_gaussian(origin, 5, 0, 0, 0)
        with pytest.raises(IndexError, match="out of range"):
            simple_cgto.primitive_gaussian(origin, -1, 0, 0, 0)

    def test_primitive_gaussian_angular_mismatch(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test primitive_gaussian raises ValueError when lx+ly+lz != l."""
        with pytest.raises(ValueError, match="must equal l"):
            simple_cgto.primitive_gaussian(origin, 0, 1, 0, 0)  # l=0 but lx=1

    def test_primitive_gaussian_p_orbital(self, origin: CartesianCoordinates):
        """Test primitive Gaussian for p-orbital at origin is zero."""
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=1,
            exponents=[1.0],
            contractions=[1.0],
        )
        # p_x at center: x^1 * ... = 0
        assert cgto.primitive_gaussian(origin, 0, 1, 0, 0) == 0.0
        # p_x off-center: should be non-zero
        off_center = CartesianCoordinates(0.5, 0.0, 0.0)
        assert cgto.primitive_gaussian(off_center, 0, 1, 0, 0) != 0.0

    def test_basis_function_s_orbital(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test basis_function for single-primitive s-orbital."""
        bf_value = simple_cgto.basis_function(origin, 0, 0, 0)
        prim_value = simple_cgto.primitive_gaussian(origin, 0, 0, 0, 0)
        # With contraction coefficient of 1.0, they should be equal
        assert abs(bf_value - 1.0 * prim_value) < 1e-10

    def test_basis_function_angular_mismatch(
        self, simple_cgto: ContractedGaussianTypeOrbital, origin: CartesianCoordinates
    ):
        """Test basis_function raises ValueError when lx+ly+lz != l."""
        with pytest.raises(ValueError, match="must equal l"):
            simple_cgto.basis_function(origin, 1, 0, 0)

    def test_basis_function_contracted(self, origin: CartesianCoordinates):
        """Test basis_function correctly sums over primitives."""
        # Create a two-primitive CGTO
        cgto = ContractedGaussianTypeOrbital(
            center=origin,
            l=0,
            exponents=[1.0, 0.5],
            contractions=[0.6, 0.4],
        )
        bf_value = cgto.basis_function(origin, 0, 0, 0)
        # Manual calculation
        prim0 = cgto.primitive_gaussian(origin, 0, 0, 0, 0)
        prim1 = cgto.primitive_gaussian(origin, 1, 0, 0, 0)
        expected = 0.6 * prim0 + 0.4 * prim1
        assert abs(bf_value - expected) < 1e-10


# ======================================================================
# Atom.make_contracted_gaussian_type_orbital Tests
# ======================================================================


class TestAtomMakeCGTO:
    """Tests for Atom.make_contracted_gaussian_type_orbital."""

    def test_populates_list(self, hydrogen_atom: Atom):
        """Test that method populates contracted_gaussian_type_orbitals."""
        assert hydrogen_atom.contracted_gaussian_type_orbitals is None
        hydrogen_atom.make_contracted_gaussian_type_orbital()
        assert hydrogen_atom.contracted_gaussian_type_orbitals is not None
        assert len(hydrogen_atom.contracted_gaussian_type_orbitals) > 0

    def test_returns_none(self, hydrogen_atom: Atom):
        """Test that method returns None."""
        result = hydrogen_atom.make_contracted_gaussian_type_orbital()
        assert result is None

    def test_atom_index_none_when_not_provided(self, hydrogen_atom: Atom):
        """Test that atom_index is None when not provided."""
        hydrogen_atom.make_contracted_gaussian_type_orbital()
        for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
            assert cgto.atom_index is None

    def test_atom_index_set_when_provided(self, hydrogen_atom: Atom):
        """Test that atom_index is set when provided."""
        hydrogen_atom.make_contracted_gaussian_type_orbital(atom_index=5)
        for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
            assert cgto.atom_index == 5

    def test_missing_basis_set(self, origin: CartesianCoordinates):
        """Test that method handles missing basis set gracefully."""
        atom = Atom(atomic_number=1, coordinates=origin)
        atom.make_contracted_gaussian_type_orbital()
        assert atom.contracted_gaussian_type_orbitals == []

    def test_missing_coordinates(self, basis_631g: Pople):
        """Test that method handles missing coordinates gracefully."""
        atom = Atom(atomic_number=1, basis_set=basis_631g)
        atom.make_contracted_gaussian_type_orbital()
        assert atom.contracted_gaussian_type_orbitals == []

    def test_missing_both(self):
        """Test that method handles missing basis set and coordinates."""
        atom = Atom(atomic_number=1)
        atom.make_contracted_gaussian_type_orbital()
        assert atom.contracted_gaussian_type_orbitals == []

    def test_center_is_cartesian_coordinates(self, hydrogen_atom: Atom):
        """Test that CGTO center is CartesianCoordinates instance."""
        hydrogen_atom.make_contracted_gaussian_type_orbital()
        for cgto in hydrogen_atom.contracted_gaussian_type_orbitals:
            assert isinstance(cgto.center, CartesianCoordinates)

    def test_hydrogen_631g_shell_count(self, hydrogen_atom: Atom):
        """Test hydrogen with 6-31G has expected number of shells."""
        hydrogen_atom.make_contracted_gaussian_type_orbital()
        # 6-31G for H: 1s (3 primitives) + 1s' (1 primitive) = 2 shells
        assert len(hydrogen_atom.contracted_gaussian_type_orbitals) == 2

    def test_oxygen_631g_shell_count(self, basis_631g: Pople, origin: CartesianCoordinates):
        """Test oxygen with 6-31G has expected number of shells."""
        oxygen = Atom(atomic_number=8, basis_set=basis_631g, coordinates=origin)
        oxygen.make_contracted_gaussian_type_orbital()
        # 6-31G for O: 1s + 2s + 2p + 2s' + 2p' = 5 shells
        assert len(oxygen.contracted_gaussian_type_orbitals) == 5


# ======================================================================
# Molecule.make_contracted_gaussian_type_orbital Tests
# ======================================================================


class TestMoleculeMakeCGTO:
    """Tests for Molecule.make_contracted_gaussian_type_orbital."""

    def test_populates_list(self, water_molecule: Molecule):
        """Test that method populates contracted_gaussian_type_orbitals."""
        assert water_molecule.contracted_gaussian_type_orbitals is None
        water_molecule.make_contracted_gaussian_type_orbital()
        assert water_molecule.contracted_gaussian_type_orbitals is not None
        assert len(water_molecule.contracted_gaussian_type_orbitals) > 0

    def test_returns_none(self, water_molecule: Molecule):
        """Test that method returns None."""
        result = water_molecule.make_contracted_gaussian_type_orbital()
        assert result is None

    def test_n_basis_matches_sum_of_n_functions(self, water_molecule: Molecule):
        """Test that n_basis equals sum of n_functions over all CGTOs."""
        water_molecule.make_contracted_gaussian_type_orbital()
        total_functions = sum(
            cgto.n_functions
            for cgto in water_molecule.contracted_gaussian_type_orbitals
        )
        assert total_functions == water_molecule.n_basis

    def test_atom_indices_correct(self, water_molecule: Molecule):
        """Test that each CGTO has correct atom_index."""
        water_molecule.make_contracted_gaussian_type_orbital()
        # Water: O (index 0), H (index 1), H (index 2)
        seen_indices = set()
        for cgto in water_molecule.contracted_gaussian_type_orbitals:
            seen_indices.add(cgto.atom_index)
        assert seen_indices == {0, 1, 2}

    def test_water_631g_shell_count(self, water_molecule: Molecule):
        """Test water with 6-31G has expected number of shells."""
        water_molecule.make_contracted_gaussian_type_orbital()
        # O: 5 shells, H: 2 shells each
        # Total: 5 + 2 + 2 = 9 shells
        assert len(water_molecule.contracted_gaussian_type_orbitals) == 9

    def test_water_631g_n_basis(self, water_molecule: Molecule):
        """Test water with 6-31G has 13 basis functions."""
        # O: 1s(1) + 2s(1) + 2p(3) + 2s'(1) + 2p'(3) = 9
        # H: 1s(1) + 1s'(1) = 2 each
        # Total: 9 + 2 + 2 = 13
        assert water_molecule.n_basis == 13

    def test_also_populates_atom_cgtos(self, water_molecule: Molecule):
        """Test that method also populates CGTOs on individual atoms."""
        water_molecule.make_contracted_gaussian_type_orbital()
        for atom in water_molecule.atoms:
            assert atom.contracted_gaussian_type_orbitals is not None
            assert len(atom.contracted_gaussian_type_orbitals) > 0


# ======================================================================
# Molecule.molecular_orbital Tests
# ======================================================================


class TestMoleculeMolecularOrbital:
    """Tests for Molecule.molecular_orbital method."""

    @staticmethod
    def _cartesian_components(l: int) -> List[Tuple[int, int, int]]:
        """Generate Cartesian angular momentum tuples for given l."""
        components = []
        for lx in range(l, -1, -1):
            for ly in range(l - lx, -1, -1):
                lz = l - lx - ly
                components.append((lx, ly, lz))
        return components

    @staticmethod
    def _build_angular_components(
        cgtos: List[ContractedGaussianTypeOrbital],
    ) -> List[Tuple[int, int, int]]:
        """Build angular components list for all CGTOs."""
        angular = []
        for cgto in cgtos:
            angular.extend(TestMoleculeMolecularOrbital._cartesian_components(cgto.l))
        return angular

    def test_raises_if_cgtos_not_built(self, water_molecule: Molecule):
        """Test that method raises if CGTOs haven't been built."""
        r = CartesianCoordinates(0.0, 0.0, 0.0)
        coeffs = [1.0] * water_molecule.n_basis
        angular = [(0, 0, 0)] * water_molecule.n_basis
        with pytest.raises(ValueError, match="CGTOs not built"):
            water_molecule.molecular_orbital(r, coeffs, angular)

    def test_raises_on_coefficients_length_mismatch(self, water_molecule: Molecule):
        """Test that method raises on coefficient length mismatch."""
        water_molecule.make_contracted_gaussian_type_orbital()
        r = CartesianCoordinates(0.0, 0.0, 0.0)
        angular = self._build_angular_components(
            water_molecule.contracted_gaussian_type_orbitals
        )
        coeffs = [1.0]  # Too short
        with pytest.raises(ValueError, match="coefficients length"):
            water_molecule.molecular_orbital(r, coeffs, angular)

    def test_raises_on_angular_length_mismatch(self, water_molecule: Molecule):
        """Test that method raises on angular_components length mismatch."""
        water_molecule.make_contracted_gaussian_type_orbital()
        r = CartesianCoordinates(0.0, 0.0, 0.0)
        coeffs = [1.0] * water_molecule.n_basis
        angular = [(0, 0, 0)]  # Too short
        with pytest.raises(ValueError, match="angular_components length"):
            water_molecule.molecular_orbital(r, coeffs, angular)

    def test_unit_vector_first_basis(self, water_molecule: Molecule):
        """Test MO evaluation with unit vector selecting first basis function."""
        water_molecule.make_contracted_gaussian_type_orbital()
        r = CartesianCoordinates(0.0, 0.0, 0.1173)  # Near oxygen
        angular = self._build_angular_components(
            water_molecule.contracted_gaussian_type_orbitals
        )
        coeffs = [1.0] + [0.0] * (water_molecule.n_basis - 1)
        
        mo_value = water_molecule.molecular_orbital(r, coeffs, angular)
        
        # Should equal the first basis function value
        first_cgto = water_molecule.contracted_gaussian_type_orbitals[0]
        first_angular = angular[0]
        expected = first_cgto.basis_function(r, *first_angular)
        assert abs(mo_value - expected) < 1e-10

    def test_zero_coefficients_gives_zero(self, water_molecule: Molecule):
        """Test that all-zero coefficients give zero MO value."""
        water_molecule.make_contracted_gaussian_type_orbital()
        r = CartesianCoordinates(0.0, 0.0, 0.0)
        angular = self._build_angular_components(
            water_molecule.contracted_gaussian_type_orbitals
        )
        coeffs = [0.0] * water_molecule.n_basis
        
        mo_value = water_molecule.molecular_orbital(r, coeffs, angular)
        assert mo_value == 0.0

    def test_linearity(self, water_molecule: Molecule):
        """Test MO evaluation is linear in coefficients."""
        water_molecule.make_contracted_gaussian_type_orbital()
        r = CartesianCoordinates(0.5, 0.3, 0.2)
        angular = self._build_angular_components(
            water_molecule.contracted_gaussian_type_orbitals
        )
        
        # Compute with coefficients a and b
        coeffs_a = [1.0] + [0.0] * (water_molecule.n_basis - 1)
        coeffs_b = [0.0, 1.0] + [0.0] * (water_molecule.n_basis - 2)
        
        mo_a = water_molecule.molecular_orbital(r, coeffs_a, angular)
        mo_b = water_molecule.molecular_orbital(r, coeffs_b, angular)
        
        # Sum should equal evaluation with summed coefficients
        coeffs_sum = [1.0, 1.0] + [0.0] * (water_molecule.n_basis - 2)
        mo_sum = water_molecule.molecular_orbital(r, coeffs_sum, angular)
        
        assert abs(mo_sum - (mo_a + mo_b)) < 1e-10
