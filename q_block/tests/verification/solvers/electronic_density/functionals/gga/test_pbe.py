"""Unit tests for compute.solvers.electronic_density.functionals.gga.pbe module.

Tests cover:
- compute_exc_vxc output shapes match input density shape
- compute_exc_vxc raises ValueError when gradient info is missing
- Energy density (exc) is negative for finite density
- Potential arrays (vxc_alpha, vxc_beta) are finite
- Gradient influence: non-zero gamma reduces or changes exc vs. zero gradient
- Spin-symmetry: equal alpha/beta densities give equal vxc_alpha and vxc_beta
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import PBE


# ======================================================================
# compute_exc_vxc — output shapes
# ======================================================================


def test_pbe_exc_vxc_output_shapes() -> None:
    """compute_exc_vxc returns three arrays each of shape (n_points,)."""
    func = PBE()
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


def test_pbe_raises_without_gamma_aa() -> None:
    """compute_exc_vxc raises ValueError when gamma_aa is not provided."""
    func = PBE()
    rho_a = np.array([0.5])
    rho_b = np.array([0.5])
    with pytest.raises(ValueError, match="gradient"):
        func.compute_exc_vxc(rho_a, rho_b)


def test_pbe_raises_without_gamma_bb() -> None:
    """compute_exc_vxc raises ValueError when gamma_bb is not provided."""
    func = PBE()
    rho_a = np.array([0.5])
    rho_b = np.array([0.5])
    with pytest.raises(ValueError, match="gradient"):
        func.compute_exc_vxc(rho_a, rho_b, gamma_aa=np.array([0.0]))


# ======================================================================
# compute_exc_vxc — energy density sign
# ======================================================================


def test_pbe_exc_negative_for_finite_density() -> None:
    """XC energy density exc is negative everywhere for positive rho."""
    func = PBE()
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


def test_pbe_potentials_are_finite() -> None:
    """vxc_alpha and vxc_beta are finite for positive densities."""
    func = PBE()
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
# compute_exc_vxc — gradient influence
# ======================================================================


def test_pbe_gradient_affects_exc() -> None:
    """Non-zero gradient changes exc compared to zero-gradient case."""
    func = PBE()
    rho_a = np.array([0.3])
    rho_b = np.array([0.3])
    zero = np.array([0.0])
    nonzero = np.array([0.5])
    exc_zero, _, _ = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=zero, gamma_ab=zero, gamma_bb=zero
    )
    exc_nz, _, _ = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=nonzero, gamma_ab=zero, gamma_bb=nonzero
    )
    assert not np.isclose(exc_zero[0], exc_nz[0])


# ======================================================================
# compute_exc_vxc — spin symmetry
# ======================================================================


def test_pbe_spin_symmetry_equal_densities() -> None:
    """vxc_alpha == vxc_beta when rho_alpha == rho_beta and gamma_aa == gamma_bb."""
    func = PBE()
    rho_a = np.array([0.1, 0.5, 1.0])
    rho_b = rho_a.copy()
    gamma = np.array([0.0, 0.01, 0.05])
    exc, vxc_a, vxc_b = func.compute_exc_vxc(
        rho_a, rho_b, gamma_aa=gamma, gamma_ab=gamma, gamma_bb=gamma
    )
    assert np.allclose(vxc_a, vxc_b, atol=1e-12)
