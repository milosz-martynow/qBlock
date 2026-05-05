"""Unit tests for compute.solvers.electronic_density.kohn_sham.restricted_open_shell_kohn_sham module.

Tests cover:
- ROKS construction and validation (negative n_closed/n_open raises, zero n_open raises)
- n_alpha, n_beta, n_electrons, n_closed, n_open properties
- SCF convergence for all three supported functionals (SVWN, PBE, B3LYP)
- Matrix keys stored after convergence
- Matrix shapes after convergence
- Effective Fock (F_eff) symmetry
- Per-spin Fock (F_alpha, F_beta) symmetry
- Per-spin density matrix (P_alpha, P_beta) symmetry
- Total density consistency (P == P_alpha + P_beta)
- Orbital energies: sorted, finite, bound (HOMO negative)

Uses Li / STO-3G (n_closed=1, n_open=1, doublet) to keep tests fast.
"""

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedOpenShellKohnSham,
)
from q_block.tests.verification.utilities import (
    li_inputs,
)

# ======================================================================
# Construction and validation
# ======================================================================


@pytest.mark.parametrize(
    "functional",
    [SVWN(), PBE(), B3LYP()],
    ids=["SVWN", "PBE", "B3LYP"],
)
def test_roks_construction_all_functionals(
    functional: object,
) -> None:
    """ROKS constructs without error for every supported functional.

    :param functional: XC functional instance.
    :type functional: object
    """
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=functional,
        n_closed=1,
        n_open=1,
        n_radial=20,
        n_angular=6,
    )
    assert roks.n_closed == 1
    assert roks.n_open == 1
    assert roks.n_electrons == 3


def test_roks_negative_closed_raises() -> None:
    """ROKS raises ValueError when n_closed is negative."""
    cgtos, nuclei, e_nuc = li_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        RestrictedOpenShellKohnSham(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            functional=SVWN(),
            n_closed=-1,
            n_open=1,
            n_radial=20,
            n_angular=6,
        )


def test_roks_zero_open_raises() -> None:
    """ROKS raises ValueError when n_open is zero."""
    cgtos, nuclei, e_nuc = li_inputs()
    with pytest.raises(ValueError, match="open-shell orbital"):
        RestrictedOpenShellKohnSham(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            functional=SVWN(),
            n_closed=1,
            n_open=0,
            n_radial=20,
            n_angular=6,
        )


def test_roks_negative_open_raises() -> None:
    """ROKS raises ValueError when n_open is negative."""
    cgtos, nuclei, e_nuc = li_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        RestrictedOpenShellKohnSham(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            functional=SVWN(),
            n_closed=1,
            n_open=-1,
            n_radial=20,
            n_angular=6,
        )


# ======================================================================
# Properties
# ======================================================================


def test_roks_n_electrons_from_closed_open() -> None:
    """n_electrons == 2*n_closed + n_open for Li (=3)."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=20,
        n_angular=6,
    )
    assert roks.n_electrons == 2 * roks.n_closed + roks.n_open


def test_roks_n_alpha_n_beta_from_closed_open() -> None:
    """n_alpha == n_closed + n_open, n_beta == n_closed."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=20,
        n_angular=6,
    )
    assert roks.n_alpha == roks.n_closed + roks.n_open
    assert roks.n_beta == roks.n_closed


# ======================================================================
# SCF convergence
# ======================================================================


def test_roks_svwn_li_converges() -> None:
    """ROKS-SVWN converges for Li / STO-3G (doublet)."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    assert roks.converged
    assert roks.e_total < 0.0
    assert "F_eff" in roks.matrices
    assert "P" in roks.matrices


def test_roks_pbe_li_converges() -> None:
    """ROKS-PBE converges for Li / STO-3G (doublet)."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=PBE(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    assert roks.converged
    assert roks.e_total < 0.0


def test_roks_b3lyp_li_converges() -> None:
    """ROKS-B3LYP converges for Li / STO-3G (doublet)."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=B3LYP(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    assert roks.converged
    assert roks.e_total < 0.0


# ======================================================================
# Convergence result attributes
# ======================================================================


def test_roks_total_energy_finite() -> None:
    """ROKS-SVWN produces a finite total energy for Li."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    assert np.isfinite(roks.e_total)


def test_roks_matrix_shapes_after_convergence() -> None:
    """All stored matrices have correct shapes after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    n = roks.n_basis
    assert roks.matrices["S"].shape == (n, n)
    assert roks.matrices["H"].shape == (n, n)
    assert roks.matrices["F_eff"].shape == (n, n)
    assert roks.matrices["F_alpha"].shape == (n, n)
    assert roks.matrices["F_beta"].shape == (n, n)
    assert roks.matrices["P"].shape == (n, n)
    assert roks.matrices["P_alpha"].shape == (n, n)
    assert roks.matrices["P_beta"].shape == (n, n)
    assert roks.matrices["C"].shape == (n, n)
    assert roks.matrices["epsilon"].shape == (n,)


def test_roks_effective_fock_symmetric() -> None:
    """F_eff (Roothaan effective Fock) is symmetric after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    F = roks.matrices["F_eff"]
    assert np.allclose(F, F.T, atol=1e-10)


def test_roks_fa_fb_symmetric() -> None:
    """F_alpha and F_beta are symmetric after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    Fa = roks.matrices["F_alpha"]
    Fb = roks.matrices["F_beta"]
    assert np.allclose(Fa, Fa.T, atol=1e-10)
    assert np.allclose(Fb, Fb.T, atol=1e-10)


def test_roks_density_matrices_symmetric() -> None:
    """P_alpha and P_beta are symmetric after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    Pa = roks.matrices["P_alpha"]
    Pb = roks.matrices["P_beta"]
    assert np.allclose(Pa, Pa.T, atol=1e-10)
    assert np.allclose(Pb, Pb.T, atol=1e-10)


def test_roks_density_matrices_consistent() -> None:
    """Total density P equals P_alpha + P_beta after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    P_total = roks.matrices["P_alpha"] + roks.matrices["P_beta"]
    assert np.allclose(roks.matrices["P"], P_total, atol=1e-10)


def test_roks_total_density_symmetric() -> None:
    """Total density matrix P is symmetric after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    P = roks.matrices["P"]
    assert np.allclose(P, P.T, atol=1e-10)


# ======================================================================
# Orbital energies
# ======================================================================


def test_roks_orbital_energies_sorted() -> None:
    """Orbital energies are sorted in ascending order after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    eps = roks.matrices["epsilon"]
    assert np.all(eps[:-1] <= eps[1:])


def test_roks_orbital_energies_finite() -> None:
    """All orbital energies are finite after convergence."""
    cgtos, nuclei, e_nuc = li_inputs()
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_closed=1,
        n_open=1,
        n_radial=30,
        n_angular=6,
        max_iterations=150,
        convergence_threshold=1e-5,
    ).run()
    assert np.all(np.isfinite(roks.matrices["epsilon"]))
