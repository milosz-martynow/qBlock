"""Unit tests for compute.solvers.electronic_density.kohn_sham.restricted_kohn_sham module.

Tests cover:
- RKS construction and validation (even-electron requirement)
- n_occ, n_alpha, n_beta, n_electrons, _shared_spin properties
- SCF convergence for all three supported functionals (SVWN, PBE, B3LYP)
- Matrix keys stored after convergence
- Matrix shapes after convergence
- Fock and density matrix symmetry
- Density trace equals electron count
- Density matrix positive-semidefiniteness
- Orbital energies: sorted, finite, HOMO negative

Uses H₂ / STO-3G to keep tests fast.
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
# Construction and validation
# ======================================================================


@pytest.mark.parametrize(
    "functional",
    [SVWN(), PBE(), B3LYP()],
    ids=["SVWN", "PBE", "B3LYP"],
)
def test_rks_construction_all_functionals(
    functional: object,
) -> None:
    """RKS constructs without error for every supported functional.

    :param functional: XC functional instance.
    :type functional: object
    """
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=functional,
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.n_electrons == 2
    assert rks.n_alpha == rks.n_beta == rks.n_occ == 1


def test_rks_odd_electrons_raises() -> None:
    """RKS raises ValueError when n_electrons is odd."""
    cgtos = h2_inputs()
    with pytest.raises(ValueError, match="even number"):
        RestrictedKohnSham(
            cgtos=cgtos,
            functional=SVWN(),
            n_electrons=1,
            n_radial=20,
            n_angular=6,
        )


# ======================================================================
# Properties
# ======================================================================


def test_rks_n_electrons_property() -> None:
    """n_electrons equals the supplied electron count."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.n_electrons == 2


def test_rks_n_occ_is_half_n_electrons() -> None:
    """n_occ equals n_electrons // 2."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.n_occ == 1


def test_rks_alpha_beta_equal_n_occ() -> None:
    """n_alpha == n_beta == n_occ for a closed-shell system."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.n_alpha == rks.n_occ
    assert rks.n_beta == rks.n_occ


def test_rks_shared_spin_is_true() -> None:
    """_shared_spin is True for RKS (closed-shell)."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks._shared_spin is True


# ======================================================================
# SCF convergence
# ======================================================================


def test_rks_svwn_h2_converges() -> None:
    """RKS-SVWN converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0
    assert "C" in rks.matrices
    assert "P" in rks.matrices


def test_rks_pbe_h2_converges() -> None:
    """RKS-PBE converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=PBE(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0


def test_rks_b3lyp_h2_converges() -> None:
    """RKS-B3LYP converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=B3LYP(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0


# ======================================================================
# Convergence result attributes
# ======================================================================


def test_rks_total_energy_finite() -> None:
    """RKS-SVWN produces a finite total energy for H₂."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert np.isfinite(rks.e_total)


def test_rks_matrix_shapes_after_convergence() -> None:
    """All stored matrices have correct shapes after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    n = rks.n_basis
    assert rks.matrices["C"].shape == (n, n)
    assert rks.matrices["epsilon"].shape == (n,)
    assert rks.matrices["F"].shape == (n, n)
    assert rks.matrices["P"].shape == (n, n)
    assert rks.matrices["S"].shape == (n, n)
    assert rks.matrices["H"].shape == (n, n)


def test_rks_fock_symmetric_after_convergence() -> None:
    """Fock matrix F is symmetric after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    F = rks.matrices["F"]
    assert np.allclose(F, F.T, atol=1e-10)


def test_rks_density_matrix_symmetric() -> None:
    """Density matrix P is symmetric after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    P = rks.matrices["P"]
    assert np.allclose(P, P.T, atol=1e-10)


def test_rks_density_matrix_positive_semidefinite() -> None:
    """Density matrix P has non-negative eigenvalues after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    eigvals = np.linalg.eigvalsh(rks.matrices["P"])
    assert np.all(eigvals >= -1e-10)


def test_rks_density_trace_equals_n_electrons() -> None:
    """tr(P @ S) equals n_electrons after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    n_elec = np.trace(rks.matrices["P"] @ rks.matrices["S"])
    assert abs(n_elec - 2.0) < 0.01


# ======================================================================
# Orbital energies
# ======================================================================


def test_rks_orbital_energies_sorted() -> None:
    """Orbital energies are sorted in ascending order after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    eps = rks.matrices["epsilon"]
    assert np.all(eps[:-1] <= eps[1:])


def test_rks_orbital_energies_finite() -> None:
    """All orbital energies are finite after convergence."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert np.all(np.isfinite(rks.matrices["epsilon"]))


def test_rks_homo_energy_negative() -> None:
    """HOMO orbital energy is negative (bound state)."""
    cgtos = h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    homo = rks.matrices["epsilon"][rks.n_occ - 1]
    assert homo < 0.0
