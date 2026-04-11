"""Unit tests for compute.solvers.wavefunction.hartree_fock.restricted_open_shell_hartree_fock module.

Tests cover:
- ROHF construction and orbital-count validation
- n_closed and n_open properties
- Electron counts derived from orbital occupations
- Result attributes before run()
- Full SCF convergence on H2/STO-3G (closed-shell, n_open=0)
- Electronic energy is negative
- Matrix keys populated after run()
- Effective Fock matrix is stored
- Physical per-spin Fock matrices are stored
- Per-spin and total density traces
- Density matrix symmetry
- MO coefficients orthonormality

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from compute import Molecule
from compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from compute.solvers.wavefunction.hartree_fock.restricted_open_shell_hartree_fock import (
    RestrictedOpenShellHartreeFock,
)
from tests.unit_tests.utilities import (
    extract_nuclei,
    h2_molecule,
    h2_rohf,
    water_molecule,
    water_rohf,
)

# ======================================================================
# Construction & Validation Tests
# ======================================================================


def test_valid_construction(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """ROHF with valid orbital counts should construct without error.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.n_closed == 1
    assert h2_rohf.n_open == 0


@pytest.mark.parametrize(
    "n_closed, n_open",
    [(-1, 0), (0, -1), (-2, -3)],
    ids=["neg-closed", "neg-open", "both-neg"],
)
def test_negative_orbital_counts_raises(
    h2_molecule: Molecule, n_closed: int, n_open: int
) -> None:
    """ROHF with negative orbital counts should raise ValueError.

    Negative orbital occupations are physically meaningless and
    must be rejected at construction.

    :param h2_molecule: Pytest fixture providing an H2 Molecule.
    :type h2_molecule: Molecule
    :param n_closed: Closed orbital count, provided by parametrize.
    :type n_closed: int
    :param n_open: Open orbital count, provided by parametrize.
    :type n_open: int
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    with pytest.raises(ValueError, match="non-negative"):
        RestrictedOpenShellHartreeFock(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            n_closed=n_closed,
            n_open=n_open,
        )


# ======================================================================
# Property Tests
# ======================================================================


def test_n_alpha_from_orbitals(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """n_alpha should equal n_closed + n_open.

    Alpha electrons occupy all closed and all open spatial orbitals.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.n_alpha == h2_rohf.n_closed + h2_rohf.n_open


def test_n_beta_equals_n_closed(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """n_beta should equal n_closed.

    Beta electrons occupy only the closed-shell (doubly-occupied)
    orbitals.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.n_beta == h2_rohf.n_closed


def test_n_electrons_property(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """n_electrons should return 2*n_closed + n_open.

    Total electrons = alpha + beta = (n_closed + n_open) + n_closed.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    expected = 2 * h2_rohf.n_closed + h2_rohf.n_open
    assert h2_rohf.n_electrons == expected


def test_shared_spin_is_false(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """_shared_spin should be False for ROHF.

    Even though ROHF shares spatial orbitals, alpha and beta
    densities generally differ (different occupation numbers), so
    _shared_spin must be False.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf._shared_spin is False


# ======================================================================
# Result Attributes Before run()
# ======================================================================


def test_converged_none_before_run(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """Before run(), converged should be None.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.converged is None


def test_e_total_none_before_run(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """Before run(), e_total should be None.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.e_total is None


def test_matrices_empty_before_run(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """Before run(), matrices dict should be empty.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    assert h2_rohf.matrices == {}


# ======================================================================
# SCF Run Tests — H2 / STO-3G (closed-shell ROHF)
# ======================================================================


def test_h2_rohf_converges(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """H2 / STO-3G ROHF (closed-shell) should converge.

    With zero open-shell orbitals, ROHF behaves like RHF and must
    converge on the textbook H2 system.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    assert h2_rohf.converged is True


def test_h2_rohf_returns_self(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """run() should return self for method chaining.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    result = h2_rohf.run()
    assert result is h2_rohf


def test_h2_rohf_iterations_positive(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Number of iterations should be a positive integer.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    assert h2_rohf.n_iterations is not None
    assert h2_rohf.n_iterations > 0


def test_h2_rohf_electronic_energy_negative(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Electronic energy should be negative for a bound system.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    assert h2_rohf.e_electronic is not None
    assert h2_rohf.e_electronic < 0.0


# ======================================================================
# Matrix Tests After Run
# ======================================================================


def test_h2_rohf_matrices_keys(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """After run(), matrices dict should contain expected ROHF keys.

    ROHF stores S, H, F_eff, P, F_alpha, F_beta, P_alpha, P_beta,
    C, epsilon — ten keys including both the effective and physical
    Fock matrices.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    expected_keys = {
        "S",
        "H",
        "F_eff",
        "P",
        "F_alpha",
        "F_beta",
        "P_alpha",
        "P_beta",
        "C",
        "epsilon",
    }
    assert expected_keys == set(h2_rohf.matrices.keys())


def test_h2_rohf_effective_fock_shape(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Effective Fock matrix should be n_basis × n_basis.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    n = h2_rohf.n_basis
    assert h2_rohf.matrices["F_eff"].shape == (n, n)


def test_h2_rohf_physical_fock_shapes(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Physical per-spin Fock matrices should be n_basis × n_basis.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    n = h2_rohf.n_basis
    assert h2_rohf.matrices["F_alpha"].shape == (n, n)
    assert h2_rohf.matrices["F_beta"].shape == (n, n)


def test_h2_rohf_physical_fock_symmetry(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Physical per-spin Fock matrices should be symmetric.

    The physical Fock matrices F_alpha and F_beta are Hermitian
    operators and must be symmetric.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    F_a = h2_rohf.matrices["F_alpha"]
    F_b = h2_rohf.matrices["F_beta"]
    np.testing.assert_allclose(F_a, F_a.T, atol=1e-10)
    np.testing.assert_allclose(F_b, F_b.T, atol=1e-10)


def test_h2_rohf_total_density_trace(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Tr(P S) should equal total electron count.

    For H2 with n_closed=1, n_open=0 the total is 2 electrons.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    P = h2_rohf.matrices["P"]
    S = h2_rohf.matrices["S"]
    np.testing.assert_allclose(np.trace(P @ S), 2.0, atol=1e-8)


def test_h2_rohf_alpha_density_trace(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Tr(P_alpha S) should equal n_alpha.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    P_a = h2_rohf.matrices["P_alpha"]
    S = h2_rohf.matrices["S"]
    np.testing.assert_allclose(
        np.trace(P_a @ S),
        float(h2_rohf.n_alpha),
        atol=1e-8,
    )


def test_h2_rohf_beta_density_trace(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Tr(P_beta S) should equal n_beta.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    P_b = h2_rohf.matrices["P_beta"]
    S = h2_rohf.matrices["S"]
    np.testing.assert_allclose(
        np.trace(P_b @ S),
        float(h2_rohf.n_beta),
        atol=1e-8,
    )


def test_h2_rohf_density_symmetry(h2_rohf: RestrictedOpenShellHartreeFock) -> None:
    """Per-spin density matrices should be symmetric.

    Density matrices are constructed as P = C_occ @ C_occ^T and
    must therefore be symmetric.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    P_a = h2_rohf.matrices["P_alpha"]
    P_b = h2_rohf.matrices["P_beta"]
    np.testing.assert_allclose(P_a, P_a.T, atol=1e-12)
    np.testing.assert_allclose(P_b, P_b.T, atol=1e-12)


def test_h2_rohf_total_density_equals_sum(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Total density P should equal P_alpha + P_beta.

    The stored total density matrix must be the sum of the per-spin
    density matrices.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    P = h2_rohf.matrices["P"]
    P_a = h2_rohf.matrices["P_alpha"]
    P_b = h2_rohf.matrices["P_beta"]
    np.testing.assert_allclose(P, P_a + P_b, atol=1e-12)


def test_h2_rohf_orbital_energies_ascending(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Orbital energies should be in ascending order.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    epsilon = h2_rohf.matrices["epsilon"]
    assert np.all(np.diff(epsilon) >= -1e-14)


def test_h2_rohf_coefficients_orthonormal(
    h2_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    r"""MO coefficients should satisfy C^T S C = I.

    ROHF uses a single set of spatial MOs that must be orthonormal
    in the S-metric.

    :param h2_rohf: Uninitialised ROHF for H2/STO-3G.
    :type h2_rohf: RestrictedOpenShellHartreeFock
    """
    h2_rohf.run()
    C = h2_rohf.matrices["C"]
    S = h2_rohf.matrices["S"]
    overlap = C.T @ S @ C
    np.testing.assert_allclose(overlap, np.eye(C.shape[1]), atol=1e-10)


# ======================================================================
# SCF Run Tests — Water / STO-3G (closed-shell ROHF)
# ======================================================================


def test_water_rohf_converges(water_rohf: RestrictedOpenShellHartreeFock) -> None:
    """Water / STO-3G ROHF (closed-shell) should converge.

    :param water_rohf: Uninitialised ROHF for water/STO-3G.
    :type water_rohf: RestrictedOpenShellHartreeFock
    """
    water_rohf.run()
    assert water_rohf.converged is True


def test_water_rohf_density_trace(
    water_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Tr(P S) for water ROHF should equal 10 electrons.

    :param water_rohf: Uninitialised ROHF for water/STO-3G.
    :type water_rohf: RestrictedOpenShellHartreeFock
    """
    water_rohf.run()
    P = water_rohf.matrices["P"]
    S = water_rohf.matrices["S"]
    np.testing.assert_allclose(np.trace(P @ S), 10.0, atol=1e-8)


def test_water_rohf_electronic_energy_negative(
    water_rohf: RestrictedOpenShellHartreeFock,
) -> None:
    """Electronic energy for water should be negative.

    :param water_rohf: Uninitialised ROHF for water/STO-3G.
    :type water_rohf: RestrictedOpenShellHartreeFock
    """
    water_rohf.run()
    assert water_rohf.e_electronic < 0.0
