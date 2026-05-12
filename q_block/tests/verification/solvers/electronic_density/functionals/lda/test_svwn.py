"""Unit tests for compute.solvers.electronic_density.functionals.lda.svwn module.

Tests cover:
- compute_exc_vxc output shapes match input density shape
- Energy density (exc) is negative for finite density (exchange-correlation)
- Potential arrays (vxc_alpha, vxc_beta) are finite
- Zero-density regions produce zero outputs (no numerical instabilities)
- Spin-symmetry: equal alpha/beta densities give equal vxc_alpha and vxc_beta
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import SVWN


# ======================================================================
# compute_exc_vxc — output shapes
# ======================================================================


def test_svwn_exc_vxc_output_shapes() -> None:
    """compute_exc_vxc returns six arrays each of shape (n_points,)."""
    func = SVWN()
    n = 50
    rho = np.linspace(0.01, 1.0, n)
    rho_a = rho / 2.0
    rho_b = rho / 2.0
    exc, vxc_a, vxc_b, h_alpha, h_ab, h_beta = func.compute_exc_vxc(rho_a, rho_b)
    assert exc.shape == (n,)
    assert vxc_a.shape == (n,)
    assert vxc_b.shape == (n,)
    assert h_alpha.shape == (n,)
    assert h_ab.shape == (n,)
    assert h_beta.shape == (n,)
    # LDA has no gradient correction — h arrays must be zero
    assert np.allclose(h_alpha, 0.0)
    assert np.allclose(h_ab, 0.0)
    assert np.allclose(h_beta, 0.0)


# ======================================================================
# compute_exc_vxc — energy density sign
# ======================================================================


def test_svwn_exc_negative_for_finite_density() -> None:
    """XC energy density exc is negative everywhere for positive rho."""
    func = SVWN()
    rho_a = np.linspace(0.01, 1.0, 100)
    rho_b = np.linspace(0.01, 1.0, 100)
    exc, _, _, _, _, _ = func.compute_exc_vxc(rho_a, rho_b)
    assert np.all(exc < 0.0)


# ======================================================================
# compute_exc_vxc — finiteness
# ======================================================================


def test_svwn_potentials_are_finite() -> None:
    """vxc_alpha and vxc_beta are finite for all positive densities."""
    func = SVWN()
    rho_a = np.linspace(1e-5, 2.0, 200)
    rho_b = np.linspace(1e-5, 2.0, 200)
    exc, vxc_a, vxc_b, h_alpha, h_ab, h_beta = func.compute_exc_vxc(rho_a, rho_b)
    assert np.all(np.isfinite(exc))
    assert np.all(np.isfinite(vxc_a))
    assert np.all(np.isfinite(vxc_b))


# ======================================================================
# compute_exc_vxc — zero density
# ======================================================================


def test_svwn_zero_density_gives_zero_output() -> None:
    """Zero-density points produce zero exc and zero potentials."""
    func = SVWN()
    rho_a = np.array([0.0, 0.0, 0.5])
    rho_b = np.array([0.0, 0.0, 0.5])
    exc, vxc_a, vxc_b, h_alpha, h_ab, h_beta = func.compute_exc_vxc(rho_a, rho_b)
    assert exc[0] == pytest.approx(0.0)
    assert exc[1] == pytest.approx(0.0)
    assert vxc_a[0] == pytest.approx(0.0)
    assert vxc_b[0] == pytest.approx(0.0)


# ======================================================================
# compute_exc_vxc — spin symmetry
# ======================================================================


def test_svwn_spin_symmetry_equal_densities() -> None:
    """vxc_alpha == vxc_beta when rho_alpha == rho_beta."""
    func = SVWN()
    rho_a = np.array([0.1, 0.5, 1.0])
    rho_b = rho_a.copy()
    exc, vxc_a, vxc_b, h_alpha, h_ab, h_beta = func.compute_exc_vxc(rho_a, rho_b)
    assert np.allclose(vxc_a, vxc_b, atol=1e-12)


# ======================================================================
# compute_exc_vxc — GGA kwargs are silently ignored
# ======================================================================


def test_svwn_accepts_and_ignores_gga_kwargs() -> None:
    """SVWN ignores extra gradient kwargs without raising."""
    func = SVWN()
    rho_a = np.array([0.5])
    rho_b = np.array([0.5])
    exc, vxc_a, vxc_b, h_alpha, h_ab, h_beta = func.compute_exc_vxc(
        rho_a,
        rho_b,
        gamma_aa=np.array([0.0]),
        gamma_ab=np.array([0.0]),
        gamma_bb=np.array([0.0]),
    )
    assert exc.shape == (1,)
