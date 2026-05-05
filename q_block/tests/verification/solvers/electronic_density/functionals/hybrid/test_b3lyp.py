"""Unit tests for compute.solvers.electronic_density.functionals.hybrid.b3lyp module.

Tests cover:
- compute_exc_vxc output shapes match input density shape
- compute_exc_vxc raises ValueError when gradient info is missing
- exact_exchange_fraction equals 0.2 (B3LYP A0 constant)
- Energy density (exc) is negative for finite density
- Potential arrays (vxc_alpha, vxc_beta) are finite
- Spin-symmetry: equal alpha/beta densities give equal vxc_alpha and vxc_beta
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import B3LYP


# ======================================================================
# exact_exchange_fraction
# ======================================================================


def test_b3lyp_exact_exchange_fraction() -> None:
    """exact_exchange_fraction is 0.20 (B3LYP A0 constant)."""
    func = B3LYP()
    assert func.exact_exchange_fraction == pytest.approx(0.20)


# ======================================================================
# compute_exc_vxc — output shapes
# ======================================================================


def test_b3lyp_exc_vxc_output_shapes() -> None:
    """compute_exc_vxc returns three arrays each of shape (n_points,)."""
    func = B3LYP()
    n = 50
    rho_a = np.linspace(0.01, 1.0, n) / 2.0
    rho_b = rho_a.copy()
    gamma = np.zeros(n)
    exc, vxc_a, vxc_b = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=gamma, gamma_ab=gamma, gamma_bb=gamma
    )
    assert exc.shape == (n,)
    assert vxc_a.shape == (n,)
    assert vxc_b.shape == (n,)


# ======================================================================
# compute_exc_vxc — raises without gradient
# ======================================================================


def test_b3lyp_raises_without_gradient() -> None:
    """compute_exc_vxc raises ValueError when gradient info is missing."""
    func = B3LYP()
    rho_a = np.array([0.5])
    rho_b = np.array([0.5])
    with pytest.raises(ValueError, match="gradient"):
        func.compute_exc_vxc(rho_a, rho_b)


# ======================================================================
# compute_exc_vxc — energy density sign
# ======================================================================


def test_b3lyp_exc_negative_for_finite_density() -> None:
    """XC energy density exc is negative everywhere for positive rho."""
    func = B3LYP()
    rho_a = np.linspace(0.01, 1.0, 100) / 2.0
    rho_b = rho_a.copy()
    gamma = np.zeros(100)
    exc, _, _ = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=gamma, gamma_ab=gamma, gamma_bb=gamma
    )
    assert np.all(exc < 0.0)


# ======================================================================
# compute_exc_vxc — finiteness
# ======================================================================


def test_b3lyp_potentials_are_finite() -> None:
    """vxc_alpha and vxc_beta are finite for positive densities."""
    func = B3LYP()
    n = 200
    rho_a = np.linspace(1e-5, 2.0, n) / 2.0
    rho_b = rho_a.copy()
    gamma = np.linspace(0.0, 0.1, n)
    exc, vxc_a, vxc_b = func.compute_exc_vxc(
        rho_a,
        rho_b,
        gamma_aa=gamma,
        gamma_ab=gamma,
        gamma_bb=gamma,
    )
    assert np.all(np.isfinite(exc))
    assert np.all(np.isfinite(vxc_a))
    assert np.all(np.isfinite(vxc_b))


# ======================================================================
# compute_exc_vxc — spin symmetry
# ======================================================================


def test_b3lyp_spin_symmetry_equal_densities() -> None:
    """vxc_alpha == vxc_beta when rho_alpha == rho_beta and gamma_aa == gamma_bb."""
    func = B3LYP()
    rho_a = np.array([0.1, 0.5, 1.0])
    rho_b = rho_a.copy()
    gamma = np.array([0.0, 0.01, 0.05])
    exc, vxc_a, vxc_b = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=gamma, gamma_ab=gamma, gamma_bb=gamma
    )
    assert np.allclose(vxc_a, vxc_b, atol=1e-12)
