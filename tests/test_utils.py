"""Unit tests for q_block.theory.utils module.

Tests cover:
- double_factorial function
- normalization_constant function for Cartesian Gaussian primitives
- get_cartesian_components function for angular momentum components
"""

import math

import pytest

from q_block.theory.utils import (
    double_factorial,
    normalization_constant,
    get_cartesian_components,
)


# ======================================================================
# double_factorial tests
# ======================================================================


class TestDoubleFactorial:
    """Tests for the double_factorial function."""

    def test_negative_one(self):
        """(-1)!! = 1 by quantum chemistry convention."""
        assert double_factorial(-1) == 1

    def test_zero(self):
        """0!! = 1."""
        assert double_factorial(0) == 1

    def test_one(self):
        """1!! = 1."""
        assert double_factorial(1) == 1

    def test_two(self):
        """2!! = 2."""
        assert double_factorial(2) == 2

    def test_three(self):
        """3!! = 3 * 1 = 3."""
        assert double_factorial(3) == 3

    def test_four(self):
        """4!! = 4 * 2 = 8."""
        assert double_factorial(4) == 8

    def test_five(self):
        """5!! = 5 * 3 * 1 = 15."""
        assert double_factorial(5) == 15

    def test_six(self):
        """6!! = 6 * 4 * 2 = 48."""
        assert double_factorial(6) == 48

    def test_seven(self):
        """7!! = 7 * 5 * 3 * 1 = 105."""
        assert double_factorial(7) == 105

    def test_returns_int(self):
        """Double factorial returns an integer."""
        assert isinstance(double_factorial(5), int)


# ======================================================================
# normalization_constant tests
# ======================================================================


class TestNormalizationConstant:
    """Tests for the normalization_constant function."""

    def test_s_orbital(self):
        """Test normalization constant for s-orbital (l=0).

        For s-orbital: N = (2*alpha/pi)^(3/4)
        """
        alpha = 1.0
        norm = normalization_constant(alpha, 0, 0, 0)
        expected = (2.0 * alpha / math.pi) ** 0.75
        assert abs(norm - expected) < 1e-10

    def test_s_orbital_different_exponent(self):
        """Test normalization constant for s-orbital with different exponent."""
        alpha = 2.5
        norm = normalization_constant(alpha, 0, 0, 0)
        expected = (2.0 * alpha / math.pi) ** 0.75
        assert abs(norm - expected) < 1e-10

    def test_p_orbital_x(self):
        """Test normalization constant for px-orbital (lx=1, ly=0, lz=0)."""
        alpha = 1.0
        norm = normalization_constant(alpha, 1, 0, 0)
        # N = (2*alpha/pi)^(3/4) * sqrt((4*alpha)^1 / 1!!)
        # 1!! = 1, (-1)!! = 1 (handled in function)
        expected = (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(4.0 * alpha)
        assert abs(norm - expected) < 1e-10

    def test_all_positive(self):
        """Test normalization constant is always positive."""
        for lx in range(4):
            for ly in range(4 - lx):
                for lz in range(4 - lx - ly):
                    for alpha in [0.5, 1.0, 2.0]:
                        norm = normalization_constant(alpha, lx, ly, lz)
                        assert norm > 0, f"Failed for lx={lx}, ly={ly}, lz={lz}, alpha={alpha}"

    def test_symmetry_in_angular_momentum(self):
        """Test that swapping angular momentum components gives same result."""
        alpha = 1.5
        norm_xy = normalization_constant(alpha, 1, 1, 0)
        norm_xz = normalization_constant(alpha, 1, 0, 1)
        norm_yz = normalization_constant(alpha, 0, 1, 1)
        # All should be equal due to symmetry
        assert abs(norm_xy - norm_xz) < 1e-10
        assert abs(norm_xy - norm_yz) < 1e-10

    def test_higher_exponent_larger_norm(self):
        """Test that higher exponent gives larger normalization constant."""
        norm_1 = normalization_constant(1.0, 0, 0, 0)
        norm_2 = normalization_constant(2.0, 0, 0, 0)
        norm_3 = normalization_constant(3.0, 0, 0, 0)
        assert norm_1 < norm_2 < norm_3

    def test_d_orbital(self):
        """Test normalization constant for d-orbital (l=2)."""
        alpha = 1.0
        # d_xy: lx=1, ly=1, lz=0
        norm = normalization_constant(alpha, 1, 1, 0)
        # N = (2*alpha/pi)^(3/4) * sqrt((4*alpha)^2 / (1!! * 1!! * (-1)!!))
        # = (2/pi)^(3/4) * sqrt(16 / 1) = (2/pi)^(3/4) * 4
        expected = (2.0 * alpha / math.pi) ** 0.75 * math.sqrt(16.0 * alpha**2)
        assert abs(norm - expected) < 1e-10


# ======================================================================
# get_cartesian_components tests
# ======================================================================


class TestGetCartesianComponents:
    """Tests for the get_cartesian_components function."""

    def test_s_orbital(self):
        """s-orbital (l=0) has one component: (0,0,0)."""
        components = get_cartesian_components(0)
        assert components == [(0, 0, 0)]

    def test_p_orbital(self):
        """p-orbital (l=1) has three components: px, py, pz."""
        components = get_cartesian_components(1)
        assert len(components) == 3
        assert (1, 0, 0) in components
        assert (0, 1, 0) in components
        assert (0, 0, 1) in components

    def test_d_orbital(self):
        """d-orbital (l=2) has six Cartesian components."""
        components = get_cartesian_components(2)
        assert len(components) == 6
        # Check all expected components
        assert (2, 0, 0) in components  # d_x2
        assert (0, 2, 0) in components  # d_y2
        assert (0, 0, 2) in components  # d_z2
        assert (1, 1, 0) in components  # d_xy
        assert (1, 0, 1) in components  # d_xz
        assert (0, 1, 1) in components  # d_yz

    def test_f_orbital(self):
        """f-orbital (l=3) has ten Cartesian components."""
        components = get_cartesian_components(3)
        assert len(components) == 10
        # Check sum of angular momentum for each
        for lx, ly, lz in components:
            assert lx + ly + lz == 3

    def test_sum_of_angular_momentum(self):
        """All components should sum to the total angular momentum l."""
        for l in range(5):
            components = get_cartesian_components(l)
            for lx, ly, lz in components:
                assert lx + ly + lz == l

    def test_number_of_components(self):
        """Number of components should be (l+1)(l+2)/2."""
        for l in range(6):
            components = get_cartesian_components(l)
            expected_count = (l + 1) * (l + 2) // 2
            assert len(components) == expected_count
