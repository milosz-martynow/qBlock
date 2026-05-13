"""Unit tests for compute.solvers.electronic_density.kohn_sham.kohn_sham module.

Tests cover the :class:`KohnSham` abstract base class, exercising
methods that are shared by all KS variants (RKS, UKS, ROKS):

- Basis function evaluation on the numerical grid (phi_grid, dphi_grid)
- Density on grid (_density_on_grid)
- Density gradient on grid (_gradient_on_grid)
- Coulomb and exchange matrix construction (_build_coulomb, _build_exchange)
- XC potential matrix construction (_build_vxc_matrix)
- XC energy computation (_compute_xc_energy)

Uses H2 / STO-3G (2 basis functions) to keep tests fast.  All base-class
methods are exercised through RestrictedKohnSham as a concrete proxy.
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
)
from q_block.tests.verification.utilities import (
    h2_inputs,
)




# ======================================================================
# Basis function grid � phi_grid
# ======================================================================


def test_phi_grid_shape() -> None:
    """phi_grid has shape (n_basis, n_points) and all entries are finite."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.phi_grid.shape == (rks.n_basis, rks.grid.n_points)
    assert np.all(np.isfinite(rks.phi_grid))


def test_dphi_grid_none_for_lda() -> None:
    """dphi_grid is None for LDA functionals (no gradient needed)."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is None


def test_dphi_grid_computed_for_gga() -> None:
    """dphi_grid has shape (3, n_basis, n_points) for GGA functionals."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=PBE(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is not None
    assert rks.dphi_grid.shape == (3, rks.n_basis, rks.grid.n_points)
    assert np.all(np.isfinite(rks.dphi_grid))


def test_dphi_grid_computed_for_hybrid() -> None:
    """dphi_grid is computed for hybrid functionals."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=B3LYP(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is not None


# ======================================================================
# _density_on_grid
# ======================================================================


def test_density_on_grid_shape_and_non_negative() -> None:
    """_density_on_grid returns (n_points,) array with rho >= 0."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    P = np.eye(rks.n_basis) * 0.5
    rho = rks._density_on_grid(P)
    assert rho.shape == (rks.grid.n_points,)
    assert np.all(rho >= 0.0)


# ======================================================================
# _gradient_on_grid
# ======================================================================


def test_gradient_on_grid_shape_and_finite() -> None:
    """_gradient_on_grid returns (3, n_points) finite array for GGA."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=PBE(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    P = np.eye(rks.n_basis) * 0.5
    grad = rks._gradient_on_grid(P)
    assert grad.shape == (3, rks.grid.n_points)
    assert np.all(np.isfinite(grad))


# ======================================================================
# _build_coulomb / _build_exchange
# ======================================================================


def test_coulomb_matrix_square_and_symmetric() -> None:
    """_build_coulomb returns a square symmetric matrix."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    n = rks.n_basis
    J = rks._build_coulomb(np.eye(n) * 0.5)
    assert J.shape == (n, n)
    assert np.allclose(J, J.T)


def test_exchange_matrix_square_and_symmetric() -> None:
    """_build_exchange returns a square symmetric matrix."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    n = rks.n_basis
    K = rks._build_exchange(np.eye(n) * 0.5)
    assert K.shape == (n, n)
    assert np.allclose(K, K.T)


# ======================================================================
# _build_vxc_matrix
# ======================================================================


def test_vxc_matrix_shape_and_symmetry_lda() -> None:
    """_build_vxc_matrix returns a square symmetric matrix for LDA."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    n = rks.n_basis
    P = np.eye(n) * 0.5
    rho_a = rks._density_on_grid(P)
    rho_b = rks._density_on_grid(P)
    _, vxc_a, vxc_b, _, _, _ = rks.functional.compute_exc_vxc(rho_a, rho_b)
    Vxc = rks._build_vxc_matrix(rho_a, rho_b, vxc_a, vxc_b)
    assert Vxc.alpha.shape == (n, n)
    assert np.allclose(Vxc.alpha, Vxc.alpha.T)


def test_vxc_matrix_shape_gga() -> None:
    """_build_vxc_matrix has correct shape for GGA functionals."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=PBE(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    n = rks.n_basis
    P = np.eye(n) * 0.5
    rho_a = rks._density_on_grid(P)
    rho_b = rks._density_on_grid(P)
    grad_a = rks._gradient_on_grid(P)
    gamma_aa = np.sum(grad_a * grad_a, axis=0)
    gamma_ab = gamma_aa
    gamma_bb = gamma_aa
    _, vxc_a, vxc_b, _, _, _ = rks.functional.compute_exc_vxc(
        rho_a,
        rho_b,
        gamma_aa=gamma_aa,
        gamma_ab=gamma_ab,
        gamma_bb=gamma_bb,
    )
    Vxc = rks._build_vxc_matrix(rho_a, rho_b, vxc_a, vxc_b)
    assert Vxc.alpha.shape == (n, n)


# ======================================================================
# _compute_xc_energy
# ======================================================================


def test_xc_energy_is_finite_float() -> None:
    """_compute_xc_energy returns a finite float for SVWN."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    P = np.eye(rks.n_basis) * 0.5
    rho_a = rks._density_on_grid(P)
    rho_b = rks._density_on_grid(P)
    rho = rho_a + rho_b
    exc, _, _, _, _, _ = rks.functional.compute_exc_vxc(rho_a, rho_b)
    e_xc = rks._compute_xc_energy(exc, rho)
    assert isinstance(e_xc, float)
    assert np.isfinite(e_xc)


