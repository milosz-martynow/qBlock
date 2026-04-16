"""Verification tests for the NumericalGrid class.

Tests that the numerical integration grid:
- Constructs with correct point counts
- Produces weights that integrate to correct values for simple functions
- Handles single-atom and multi-atom systems
- Validates input parameters
"""

import numpy as np
import pytest

from compute.models.integrals.numerical_grid import NumericalGrid


# ======================================================================
# Construction and basic properties
# ======================================================================


def test_numerical_grid_single_atom_point_count() -> None:
    """Grid for single atom has n_radial * n_angular points."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(nuclei=nuclei, n_radial=20, n_angular=6)
    assert grid.n_points == 20 * 6


def test_numerical_grid_two_atoms_point_count() -> None:
    """Grid for two atoms has 2 * n_radial * n_angular points."""
    nuclei = [
        (1, (0.0, 0.0, 0.0)),
        (1, (0.0, 0.0, 1.4)),
    ]
    grid = NumericalGrid(
        nuclei=nuclei, n_radial=20, n_angular=6
    )
    assert grid.n_points == 2 * 20 * 6


def test_numerical_grid_coords_shape() -> None:
    """Grid coords shape is (n_points, 3)."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(nuclei=nuclei, n_radial=10, n_angular=6)
    assert grid.coords.shape == (grid.n_points, 3)


def test_numerical_grid_weights_shape() -> None:
    """Grid weights shape is (n_points,)."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(nuclei=nuclei, n_radial=10, n_angular=6)
    assert grid.weights.shape == (grid.n_points,)


def test_numerical_grid_weights_positive() -> None:
    """Single-atom grid weights are non-negative."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(nuclei=nuclei, n_radial=30, n_angular=6)
    assert np.all(grid.weights >= 0.0)


# ======================================================================
# Integration accuracy
# ======================================================================


def test_numerical_grid_integrates_gaussian() -> None:
    """Numerical integration of a Gaussian yields correct value.

    Tests that :math:`\\int e^{-r^2} d^3r = \\pi^{3/2}` using a
    fine grid adequate for the Euler-Maclaurin radial quadrature.
    """
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(
        nuclei=nuclei, n_radial=100, n_angular=26
    )
    r_sq = np.sum(grid.coords ** 2, axis=1)
    f = np.exp(-r_sq)
    integral = np.sum(grid.weights * f)
    expected = np.pi ** 1.5
    assert abs(integral - expected) / expected < 0.01


# ======================================================================
# Validation
# ======================================================================


def test_numerical_grid_invalid_n_angular() -> None:
    """Invalid n_angular raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported n_angular"):
        NumericalGrid(
            nuclei=[(1, (0.0, 0.0, 0.0))],
            n_radial=10,
            n_angular=5,
        )


def test_numerical_grid_invalid_n_radial() -> None:
    """Invalid n_radial raises ValueError."""
    with pytest.raises(ValueError, match="n_radial must be"):
        NumericalGrid(
            nuclei=[(1, (0.0, 0.0, 0.0))],
            n_radial=0,
            n_angular=6,
        )


# ======================================================================
# Lebedev quadrature sizes
# ======================================================================


@pytest.mark.parametrize("n_angular", [6, 14, 26])
def test_numerical_grid_supported_angular_sizes(
    n_angular: int,
) -> None:
    """All supported angular grid sizes construct successfully."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(
        nuclei=nuclei, n_radial=10, n_angular=n_angular
    )
    assert grid.n_points == 10 * n_angular


def test_numerical_grid_repr() -> None:
    """__repr__ returns expected format."""
    nuclei = [(1, (0.0, 0.0, 0.0))]
    grid = NumericalGrid(nuclei=nuclei, n_radial=10, n_angular=6)
    r = repr(grid)
    assert "NumericalGrid" in r
    assert "n_points=60" in r
