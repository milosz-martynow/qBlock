"""Unit tests for compute.solvers.wavefunction.hartree_fock.unrestricted_hartree_fock module.

Tests cover:
- UHF construction and electron-count validation
- _shared_spin is False
- Result attributes before run()
- Full SCF convergence on H2/STO-3G (closed-shell configuration)
- Electronic energy is negative
- Per-spin matrix keys populated after run()
- Per-spin density traces
- Per-spin Fock matrix symmetry
- Per-spin orbital energies ascending
- MO coefficients orthonormality
- Full SCF convergence on water/STO-3G

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block.compute import Molecule
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.solvers.wavefunction.hartree_fock.unrestricted_hartree_fock import (
    UnrestrictedHartreeFock,
)
from q_block.tests.verification.utilities import (
    extract_nuclei,
    h2_molecule,
    h2_uhf,
    water_molecule,
    water_uhf,
)

# ======================================================================
# Construction & Validation Tests
# ======================================================================


def test_valid_construction(h2_uhf: UnrestrictedHartreeFock) -> None:
    """UHF with valid electron counts should construct without error.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.n_alpha == 1
    assert h2_uhf.n_beta == 1


@pytest.mark.parametrize(
    "n_alpha, n_beta",
    [(-1, 1), (1, -1), (-2, -3)],
    ids=["neg-alpha", "neg-beta", "both-neg"],
)
def test_negative_electrons_raises(
    h2_molecule: Molecule, n_alpha: int, n_beta: int
) -> None:
    """UHF with negative electron count should raise ValueError.

    Negative electron numbers are physically meaningless and must
    be rejected at construction.

    :param h2_molecule: Pytest fixture providing an H2 Molecule.
    :type h2_molecule: Molecule
    :param n_alpha: Alpha electron count, provided by parametrize.
    :type n_alpha: int
    :param n_beta: Beta electron count, provided by parametrize.
    :type n_beta: int
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    with pytest.raises(ValueError, match="non-negative"):
        UnrestrictedHartreeFock(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            n_alpha=n_alpha,
            n_beta=n_beta,
        )


# ======================================================================
# Property Tests
# ======================================================================


def test_shared_spin_is_false(h2_uhf: UnrestrictedHartreeFock) -> None:
    """_shared_spin should be False for UHF.

    UHF uses independent alpha and beta orbitals, so the spin
    channels are never shared.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf._shared_spin is False


def test_n_electrons_property(h2_uhf: UnrestrictedHartreeFock) -> None:
    """n_electrons should return the total electron count.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.n_electrons == 2


# ======================================================================
# Result Attributes Before run()
# ======================================================================


def test_converged_none_before_run(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Before run(), converged should be None.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.converged is None


def test_e_total_none_before_run(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Before run(), e_total should be None.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.e_total is None


def test_matrices_empty_before_run(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Before run(), matrices dict should be empty.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.matrices == {}


# ======================================================================
# SCF Run Tests — H2 / STO-3G
# ======================================================================


def test_h2_uhf_converges(h2_uhf: UnrestrictedHartreeFock) -> None:
    """H2 / STO-3G UHF (closed-shell) should converge.

    Even though UHF allows different alpha/beta orbitals, a
    closed-shell configuration should converge smoothly.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    assert h2_uhf.converged is True


def test_h2_uhf_returns_self(h2_uhf: UnrestrictedHartreeFock) -> None:
    """run() should return self for method chaining.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    result = h2_uhf.run()
    assert result is h2_uhf


def test_h2_uhf_iterations_positive(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Number of iterations should be a positive integer.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    assert h2_uhf.n_iterations is not None
    assert h2_uhf.n_iterations > 0


def test_h2_uhf_electronic_energy_negative(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Electronic energy should be negative for a bound system.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    assert h2_uhf.e_electronic is not None
    assert h2_uhf.e_electronic < 0.0


# ======================================================================
# Matrix Tests After Run
# ======================================================================


def test_h2_uhf_matrices_keys(h2_uhf: UnrestrictedHartreeFock) -> None:
    """After run(), matrices dict should contain expected UHF keys.

    UHF stores per-spin matrices: S, H, F_alpha, F_beta, P_alpha,
    P_beta, C_alpha, C_beta, epsilon_alpha, epsilon_beta.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    expected_keys = {
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
    }
    assert expected_keys == set(h2_uhf.matrices.keys())


def test_h2_uhf_alpha_density_trace(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Tr(P_alpha S) should equal n_alpha.

    The trace of the alpha density with the overlap matrix gives
    the alpha electron count.  For H2 UHF with n_alpha=1 this is 1.0.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    P_a = h2_uhf.matrices["P_alpha"]
    S = h2_uhf.matrices["S"]
    np.testing.assert_allclose(np.trace(P_a @ S), 1.0, atol=1e-8)


def test_h2_uhf_beta_density_trace(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Tr(P_beta S) should equal n_beta.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    P_b = h2_uhf.matrices["P_beta"]
    S = h2_uhf.matrices["S"]
    np.testing.assert_allclose(np.trace(P_b @ S), 1.0, atol=1e-8)


def test_h2_uhf_total_density_trace(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Tr((P_alpha + P_beta) S) should equal total electron count.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    P_a = h2_uhf.matrices["P_alpha"]
    P_b = h2_uhf.matrices["P_beta"]
    S = h2_uhf.matrices["S"]
    n_total = np.trace((P_a + P_b) @ S)
    np.testing.assert_allclose(n_total, 2.0, atol=1e-8)


def test_h2_uhf_fock_alpha_symmetry(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Converged alpha Fock matrix should be symmetric.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    F_a = h2_uhf.matrices["F_alpha"]
    np.testing.assert_allclose(F_a, F_a.T, atol=1e-10)


def test_h2_uhf_fock_beta_symmetry(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Converged beta Fock matrix should be symmetric.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    h2_uhf.run()
    F_b = h2_uhf.matrices["F_beta"]
    np.testing.assert_allclose(F_b, F_b.T, atol=1e-10)


@pytest.mark.parametrize(
    "key",
    ["epsilon_alpha", "epsilon_beta"],
    ids=["alpha", "beta"],
)
def test_h2_uhf_orbital_energies_ascending(
    h2_uhf: UnrestrictedHartreeFock, key: str
) -> None:
    """Per-spin orbital energies should be in ascending order.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    :param key: Matrix key for orbital energies, provided by parametrize.
    :type key: str
    """
    h2_uhf.run()
    epsilon = h2_uhf.matrices[key]
    assert np.all(np.diff(epsilon) >= -1e-14)


@pytest.mark.parametrize(
    "key",
    ["C_alpha", "C_beta"],
    ids=["alpha", "beta"],
)
def test_h2_uhf_coefficients_orthonormal(
    h2_uhf: UnrestrictedHartreeFock, key: str
) -> None:
    r"""Per-spin MO coefficients should satisfy C^T S C = I.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    :param key: Matrix key for MO coefficients, provided by parametrize.
    :type key: str
    """
    h2_uhf.run()
    C = h2_uhf.matrices[key]
    S = h2_uhf.matrices["S"]
    overlap = C.T @ S @ C
    np.testing.assert_allclose(overlap, np.eye(C.shape[1]), atol=1e-10)


# ======================================================================
# SCF Run Tests — Water / STO-3G
# ======================================================================


def test_water_uhf_converges(water_uhf: UnrestrictedHartreeFock) -> None:
    """Water / STO-3G UHF (closed-shell) should converge.

    :param water_uhf: Uninitialised UHF for water/STO-3G.
    :type water_uhf: UnrestrictedHartreeFock
    """
    water_uhf.run()
    assert water_uhf.converged is True


def test_water_uhf_density_traces(water_uhf: UnrestrictedHartreeFock) -> None:
    """Per-spin density traces for water should match electron counts.

    Water has 5 alpha and 5 beta electrons in closed-shell UHF.

    :param water_uhf: Uninitialised UHF for water/STO-3G.
    :type water_uhf: UnrestrictedHartreeFock
    """
    water_uhf.run()
    P_a = water_uhf.matrices["P_alpha"]
    P_b = water_uhf.matrices["P_beta"]
    S = water_uhf.matrices["S"]
    np.testing.assert_allclose(np.trace(P_a @ S), 5.0, atol=1e-8)
    np.testing.assert_allclose(np.trace(P_b @ S), 5.0, atol=1e-8)


def test_water_uhf_electronic_energy_negative(
    water_uhf: UnrestrictedHartreeFock,
) -> None:
    """Electronic energy for water should be negative.

    :param water_uhf: Uninitialised UHF for water/STO-3G.
    :type water_uhf: UnrestrictedHartreeFock
    """
    water_uhf.run()
    assert water_uhf.e_electronic < 0.0
