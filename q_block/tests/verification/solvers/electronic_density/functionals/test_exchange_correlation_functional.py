"""Verification tests for exchange-correlation functionals.

Tests that:
- SVWN (LDA) produces finite, sensible energy densities and potentials
- PBE (GGA) requires and uses gradient information
- B3LYP (hybrid) has correct exact exchange fraction
- All functionals handle zero-density points gracefully
- Functional type properties are correct
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
    ExchangeCorrelationFunctional,
)


# ======================================================================
# Functional type and properties
# ======================================================================


def test_svwn_type_is_lda() -> None:
    """SVWN is classified as LDA."""
    f = SVWN()
    assert f.functional_type == "lda"
    assert f.name == "SVWN"


def test_svwn_no_gradient_needed() -> None:
    """SVWN does not require density gradient."""
    f = SVWN()
    assert not f.needs_gradient


def test_svwn_no_exact_exchange() -> None:
    """SVWN has zero exact exchange fraction."""
    f = SVWN()
    assert f.exact_exchange_fraction == 0.0


def test_pbe_type_is_gga() -> None:
    """PBE is classified as GGA."""
    f = PBE()
    assert f.functional_type == "gga"
    assert f.name == "PBE"


def test_pbe_needs_gradient() -> None:
    """PBE requires density gradient."""
    f = PBE()
    assert f.needs_gradient


def test_pbe_no_exact_exchange() -> None:
    """PBE has zero exact exchange fraction."""
    f = PBE()
    assert f.exact_exchange_fraction == 0.0


def test_b3lyp_type_is_hybrid() -> None:
    """B3LYP is classified as hybrid."""
    f = B3LYP()
    assert f.functional_type == "hybrid"
    assert f.name == "B3LYP"


def test_b3lyp_needs_gradient() -> None:
    """B3LYP requires density gradient."""
    f = B3LYP()
    assert f.needs_gradient


def test_b3lyp_exact_exchange_fraction() -> None:
    """B3LYP has 20% exact exchange."""
    f = B3LYP()
    assert abs(f.exact_exchange_fraction - 0.20) < 1e-10


# ======================================================================
# SVWN (LDA) computation
# ======================================================================


def test_svwn_uniform_density() -> None:
    """SVWN returns finite values for uniform density."""
    f = SVWN()
    n_pts = 100
    rho = np.full(n_pts, 0.1)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(
        rho * 0.5, rho * 0.5
    )
    assert np.all(np.isfinite(exc))
    assert np.all(np.isfinite(vxc_a))
    assert np.all(np.isfinite(vxc_b))


def test_svwn_exchange_is_negative() -> None:
    """SVWN exchange-correlation energy density is negative."""
    f = SVWN()
    n_pts = 50
    rho = np.full(n_pts, 0.5)
    exc, _, _ = f.compute_exc_vxc(rho * 0.5, rho * 0.5)
    mask = rho > 1e-18
    assert np.all(exc[mask] < 0.0)


def test_svwn_zero_density() -> None:
    """SVWN returns zeros for zero density."""
    f = SVWN()
    n_pts = 10
    rho = np.zeros(n_pts)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(rho, rho)
    assert np.allclose(exc, 0.0)
    assert np.allclose(vxc_a, 0.0)
    assert np.allclose(vxc_b, 0.0)


def test_svwn_symmetric_closed_shell() -> None:
    """SVWN returns identical alpha/beta potentials for equal densities."""
    f = SVWN()
    n_pts = 50
    rho_half = np.linspace(0.01, 1.0, n_pts)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(rho_half, rho_half)
    np.testing.assert_allclose(vxc_a, vxc_b, atol=1e-12)


def test_svwn_output_shapes() -> None:
    """SVWN outputs have correct shapes."""
    f = SVWN()
    n_pts = 30
    rho = np.random.uniform(0.01, 1.0, n_pts)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(
        rho * 0.6, rho * 0.4
    )
    assert exc.shape == (n_pts,)
    assert vxc_a.shape == (n_pts,)
    assert vxc_b.shape == (n_pts,)


# ======================================================================
# PBE (GGA) computation
# ======================================================================


def test_pbe_requires_gradient() -> None:
    """PBE raises ValueError without gradient information."""
    f = PBE()
    rho = np.full(10, 0.1)
    with pytest.raises(ValueError, match="requires density gradient"):
        f.compute_exc_vxc(rho * 0.5, rho * 0.5)


def test_pbe_uniform_density() -> None:
    """PBE returns finite values for uniform density with gradients."""
    f = PBE()
    n_pts = 50
    rho = np.full(n_pts, 0.1)
    gamma = np.full(n_pts, 0.01)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(
        rho * 0.5,
        rho * 0.5,
        gamma_aa=gamma,
        gamma_ab=gamma * 0.5,
        gamma_bb=gamma,
    )
    assert np.all(np.isfinite(exc))
    assert np.all(np.isfinite(vxc_a))
    assert np.all(np.isfinite(vxc_b))


def test_pbe_zero_gradient_matches_lda_trends() -> None:
    """PBE with zero gradient is close to LDA exchange."""
    pbe = PBE()
    lda = SVWN()
    n_pts = 50
    rho = np.linspace(0.01, 1.0, n_pts)
    gamma_zero = np.zeros(n_pts)
    exc_pbe, _, _ = pbe.compute_exc_vxc(
        rho * 0.5,
        rho * 0.5,
        gamma_aa=gamma_zero,
        gamma_ab=gamma_zero,
        gamma_bb=gamma_zero,
    )
    exc_lda, _, _ = lda.compute_exc_vxc(rho * 0.5, rho * 0.5)
    # PBE with zero gradient should be similar to LDA
    mask = rho > 1e-18
    assert np.corrcoef(exc_pbe[mask], exc_lda[mask])[0, 1] > 0.9


# ======================================================================
# B3LYP (hybrid) computation
# ======================================================================


def test_b3lyp_requires_gradient() -> None:
    """B3LYP raises ValueError without gradient information."""
    f = B3LYP()
    rho = np.full(10, 0.1)
    with pytest.raises(ValueError, match="requires density gradient"):
        f.compute_exc_vxc(rho * 0.5, rho * 0.5)


def test_b3lyp_uniform_density() -> None:
    """B3LYP returns finite values for uniform density with gradients."""
    f = B3LYP()
    n_pts = 50
    rho = np.full(n_pts, 0.1)
    gamma = np.full(n_pts, 0.01)
    exc, vxc_a, vxc_b = f.compute_exc_vxc(
        rho * 0.5,
        rho * 0.5,
        gamma_aa=gamma,
        gamma_ab=gamma * 0.5,
        gamma_bb=gamma,
    )
    assert np.all(np.isfinite(exc))
    assert np.all(np.isfinite(vxc_a))
    assert np.all(np.isfinite(vxc_b))


def test_b3lyp_mixing_parameters() -> None:
    """B3LYP has correct mixing parameters."""
    f = B3LYP()
    assert abs(f.a0 - 0.20) < 1e-10
    assert abs(f.ax - 0.72) < 1e-10
    assert abs(f.ac - 0.81) < 1e-10


# ======================================================================
# Base class validation
# ======================================================================


def test_invalid_functional_type() -> None:
    """Invalid functional type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown functional_type"):

        class BadFunctional(ExchangeCorrelationFunctional):
            def compute_exc_vxc(
                self, rho_alpha, rho_beta, **kwargs
            ):
                pass

        BadFunctional(
            name="Bad", functional_type="invalid"
        )


# ======================================================================
# Repr
# ======================================================================


def test_svwn_repr() -> None:
    """SVWN repr contains expected information."""
    f = SVWN()
    r = repr(f)
    assert "SVWN" in r
    assert "lda" in r


def test_pbe_repr() -> None:
    """PBE repr contains expected information."""
    f = PBE()
    r = repr(f)
    assert "PBE" in r
    assert "gga" in r


def test_b3lyp_repr() -> None:
    """B3LYP repr contains expected information."""
    f = B3LYP()
    r = repr(f)
    assert "B3LYP" in r
    assert "hybrid" in r
