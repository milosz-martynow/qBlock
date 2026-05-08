"""Unit tests for compute.solvers.electronic_density.kohn_sham.unrestricted_kohn_sham module.

Tests cover:
- UKS construction and validation (non-negative electron counts)
- n_alpha, n_beta, n_electrons, _shared_spin properties
- SCF convergence for all three supported functionals (SVWN, PBE, B3LYP)
- Open-shell H atom (n_alpha=1, n_beta=0)
- Matrix keys stored after convergence
- Matrix shapes after convergence
- Fock and density matrix symmetry for both spin channels
- Density traces per spin channel
- Total density non-negative on grid
- Orbital energies: sorted, finite, alpha HOMO negative

Uses H₂ / STO-3G (closed-shell) and H atom / STO-3G (open-shell) to keep tests fast.
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    UnrestrictedKohnSham,
)
from q_block.tests.verification.utilities import (
    h2_inputs,
    h_atom_inputs,
)


# ======================================================================
# Construction and validation
# ======================================================================


@pytest.mark.parametrize(
    "functional",
    [SVWN(), PBE(), B3LYP()],
    ids=["SVWN", "PBE", "B3LYP"],
)
def test_uks_construction_all_functionals(
    functional: object,
) -> None:
    """UKS constructs without error for every supported functional.

    :param functional: XC functional instance.
    :type functional: object
    """
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=functional,
        n_alpha=1,
        n_beta=1,
        n_radial=20,
        n_angular=6,
    )
    assert uks.n_electrons == 2
    assert uks.n_alpha == 1
    assert uks.n_beta == 1


def test_uks_negative_electrons_raises() -> None:
    """UKS raises ValueError when n_alpha or n_beta is negative."""
    cgtos = h2_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        UnrestrictedKohnSham(
            cgtos=cgtos,
            functional=SVWN(),
            n_alpha=-1,
            n_beta=0,
            n_radial=20,
            n_angular=6,
        )


# ======================================================================
# Properties
# ======================================================================


def test_uks_n_electrons_is_sum_of_spins() -> None:
    """n_electrons equals n_alpha + n_beta."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=20,
        n_angular=6,
    )
    assert uks.n_electrons == uks.n_alpha + uks.n_beta


def test_uks_shared_spin_is_false() -> None:
    """_shared_spin is False for UKS (independent spin channels)."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=20,
        n_angular=6,
    )
    assert uks._shared_spin is False


# ======================================================================
# SCF convergence
# ======================================================================


def test_uks_svwn_h2_converges() -> None:
    """UKS-SVWN converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert uks.converged
    assert uks.e_total < 0.0
    assert "C_alpha" in uks.matrices
    assert "C_beta" in uks.matrices


def test_uks_pbe_h2_converges() -> None:
    """UKS-PBE converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=PBE(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert uks.converged
    assert uks.e_total < 0.0


def test_uks_b3lyp_h2_converges() -> None:
    """UKS-B3LYP converges for H₂ / STO-3G."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=B3LYP(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert uks.converged
    assert uks.e_total < 0.0


def test_uks_svwn_h_atom_converges() -> None:
    """UKS-SVWN converges for H atom (n_alpha=1, n_beta=0)."""
    cgtos = h_atom_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=0,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert uks.converged
    assert uks.e_total < 0.0
    assert uks.n_electrons == 1


# ======================================================================
# Convergence result attributes
# ======================================================================


def test_uks_total_energy_finite() -> None:
    """UKS-SVWN produces a finite total energy for H₂."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert np.isfinite(uks.e_total)


def test_uks_matrix_keys_after_convergence() -> None:
    """All expected matrix keys are present after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    expected_keys = (
        "S",
        "H",
        "F_alpha",
        "F_beta",
        "P_alpha",
        "P_beta",
        "C_alpha",
        "C_beta",
        "epsilon_alpha",
        "epsilon_beta",
    )
    for key in expected_keys:
        assert key in uks.matrices


def test_uks_matrix_shapes_after_convergence() -> None:
    """All stored matrices have correct shapes after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    n = uks.n_basis
    assert uks.matrices["C_alpha"].shape == (n, n)
    assert uks.matrices["C_beta"].shape == (n, n)
    assert uks.matrices["epsilon_alpha"].shape == (n,)
    assert uks.matrices["epsilon_beta"].shape == (n,)
    assert uks.matrices["F_alpha"].shape == (n, n)
    assert uks.matrices["F_beta"].shape == (n, n)
    assert uks.matrices["P_alpha"].shape == (n, n)
    assert uks.matrices["P_beta"].shape == (n, n)


def test_uks_density_matrices_symmetric() -> None:
    """P_alpha and P_beta are symmetric after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    Pa = uks.matrices["P_alpha"]
    Pb = uks.matrices["P_beta"]
    assert np.allclose(Pa, Pa.T, atol=1e-10)
    assert np.allclose(Pb, Pb.T, atol=1e-10)


def test_uks_fock_matrices_symmetric() -> None:
    """F_alpha and F_beta are symmetric after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    Fa = uks.matrices["F_alpha"]
    Fb = uks.matrices["F_beta"]
    assert np.allclose(Fa, Fa.T, atol=1e-10)
    assert np.allclose(Fb, Fb.T, atol=1e-10)


def test_uks_density_traces_match_spin_counts() -> None:
    """tr(P_sigma @ S) matches n_sigma for each spin channel."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    S = uks.matrices["S"]
    n_a = np.trace(uks.matrices["P_alpha"] @ S)
    n_b = np.trace(uks.matrices["P_beta"] @ S)
    assert abs(n_a - 1.0) < 0.01
    assert abs(n_b - 1.0) < 0.01


def test_uks_total_density_non_negative_on_grid() -> None:
    """Total density on grid is non-negative after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    P_total = uks.matrices["P_alpha"] + uks.matrices["P_beta"]
    rho = uks._density_on_grid(P_total)
    assert np.all(rho >= 0.0)


# ======================================================================
# Orbital energies
# ======================================================================


def test_uks_orbital_energies_sorted() -> None:
    """epsilon_alpha and epsilon_beta are sorted ascending after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    eps_a = uks.matrices["epsilon_alpha"]
    eps_b = uks.matrices["epsilon_beta"]
    assert np.all(eps_a[:-1] <= eps_a[1:])
    assert np.all(eps_b[:-1] <= eps_b[1:])


def test_uks_orbital_energies_finite() -> None:
    """All orbital energies are finite after convergence."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert np.all(np.isfinite(uks.matrices["epsilon_alpha"]))
    assert np.all(np.isfinite(uks.matrices["epsilon_beta"]))


def test_uks_homo_alpha_energy_negative() -> None:
    """Alpha HOMO orbital energy is negative (bound state)."""
    cgtos = h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    homo_a = uks.matrices["epsilon_alpha"][uks.n_alpha - 1]
    assert homo_a < 0.0
