"""Unit tests for compute.solvers.wavefunction.hartree_fock.restricted_hartree_fock module.

Tests cover:
- RHF construction and even-electron validation
- n_occ and _shared_spin properties
- Result attributes before run()
- Full SCF convergence on H2/STO-3G
- Electronic energy is negative
- Density trace equals electron count
- Density idempotency in the S-metric
- Fock matrix symmetry after convergence
- Matrix keys populated after run()
- Full SCF convergence on water/STO-3G

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block.compute import Molecule
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.solvers.wavefunction.hartree_fock.restricted_hartree_fock import (
    RestrictedHartreeFock,
)
from q_block.tests.verification.utilities import (
    extract_nuclei,
    h2_molecule,
    h2_rhf,
    water_molecule,
    water_rhf,
)

# ======================================================================
# Construction & Validation Tests
# ======================================================================


def test_even_electrons_accepted(h2_molecule: Molecule) -> None:
    """RHF with an even electron count should construct without error.

    Closed-shell RHF requires an even number of electrons.  Passing
    n_electrons=2 for H2 must succeed.

    :param h2_molecule: Pytest fixture providing an H2 Molecule.
    :type h2_molecule: Molecule
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    rhf = RestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_electrons=2,
    )
    assert rhf.n_occ == 1


@pytest.mark.parametrize(
    "n_electrons",
    [1, 3, 5, 7],
    ids=["1e", "3e", "5e", "7e"],
)
def test_odd_electrons_raises(h2_molecule: Molecule, n_electrons: int) -> None:
    """RHF with an odd electron count should raise ValueError.

    Closed-shell RHF mandates even electron counts.  Odd values
    must be rejected at construction to prevent invalid orbital
    occupations.

    :param h2_molecule: Pytest fixture providing an H2 Molecule.
    :type h2_molecule: Molecule
    :param n_electrons: Odd electron count, provided by parametrize.
    :type n_electrons: int
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    with pytest.raises(ValueError, match="even number of electrons"):
        RestrictedHartreeFock(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            n_electrons=n_electrons,
        )


# ======================================================================
# Property Tests
# ======================================================================


def test_n_occ(h2_rhf: RestrictedHartreeFock) -> None:
    """n_occ should be n_electrons // 2.

    For H2 with 2 electrons, n_occ must be 1 (one doubly-occupied
    spatial orbital).

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.n_occ == 1


def test_shared_spin_is_true(h2_rhf: RestrictedHartreeFock) -> None:
    """_shared_spin should be True for RHF.

    In closed-shell RHF, alpha and beta densities are identical
    by construction.  This property enables computational shortcuts.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf._shared_spin is True


# ======================================================================
# Result Attributes Before run()
# ======================================================================


def test_converged_none_before_run(h2_rhf: RestrictedHartreeFock) -> None:
    """Before run(), converged should be None.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.converged is None


def test_e_total_none_before_run(h2_rhf: RestrictedHartreeFock) -> None:
    """Before run(), e_total should be None.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.e_total is None


def test_matrices_empty_before_run(h2_rhf: RestrictedHartreeFock) -> None:
    """Before run(), matrices dict should be empty.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.matrices == {}


# ======================================================================
# SCF Run Tests — H2 / STO-3G
# ======================================================================


def test_h2_rhf_converges(h2_rhf: RestrictedHartreeFock) -> None:
    """H2 / STO-3G RHF should converge.

    H2 with a minimal basis set is a textbook case that must
    always converge within the default iteration limit.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    assert h2_rhf.converged is True


def test_h2_rhf_returns_self(h2_rhf: RestrictedHartreeFock) -> None:
    """run() should return self for method chaining.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    result = h2_rhf.run()
    assert result is h2_rhf


def test_h2_rhf_iterations_positive(h2_rhf: RestrictedHartreeFock) -> None:
    """Number of iterations should be a positive integer.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    assert h2_rhf.n_iterations is not None
    assert h2_rhf.n_iterations > 0


def test_h2_rhf_electronic_energy_negative(h2_rhf: RestrictedHartreeFock) -> None:
    """Electronic energy should be negative for a bound system.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    assert h2_rhf.e_electronic is not None
    assert h2_rhf.e_electronic < 0.0


# ======================================================================
# Matrix Tests After Run
# ======================================================================


def test_h2_rhf_matrices_keys(h2_rhf: RestrictedHartreeFock) -> None:
    """After run(), matrices dict should contain expected RHF keys.

    RHF stores S, H, F, P, C, epsilon — exactly six keys.
    The single-matrix representation (no per-spin split) is expected
    for closed-shell.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    expected_keys = {"S", "H", "F", "P", "C", "epsilon"}
    assert expected_keys == set(h2_rhf.matrices.keys())


def test_h2_rhf_density_trace(h2_rhf: RestrictedHartreeFock) -> None:
    """Tr(P S) should equal the number of electrons.

    The trace of the density-overlap product gives the total
    electron count.  For H2 this must be 2.0.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    P = h2_rhf.matrices["P"]
    S = h2_rhf.matrices["S"]
    n_electrons = np.trace(P @ S)
    np.testing.assert_allclose(n_electrons, 2.0, atol=1e-8)


def test_h2_rhf_density_idempotent(h2_rhf: RestrictedHartreeFock) -> None:
    r"""For RHF, P S P should equal 2 P (S-metric idempotency).

    Because P = 2 C_occ C_occ^T for closed-shell, the projective
    property PSP = 2P must hold.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    P = h2_rhf.matrices["P"]
    S = h2_rhf.matrices["S"]
    PSP = P @ S @ P
    np.testing.assert_allclose(PSP, 2.0 * P, atol=1e-8)


def test_h2_rhf_fock_symmetry(h2_rhf: RestrictedHartreeFock) -> None:
    """Converged Fock matrix should be symmetric.

    The Fock matrix is Hermitian by construction.  After convergence
    the stored matrix F must satisfy F = F^T.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    F = h2_rhf.matrices["F"]
    np.testing.assert_allclose(F, F.T, atol=1e-10)


def test_h2_rhf_orbital_energies_ascending(h2_rhf: RestrictedHartreeFock) -> None:
    """Orbital energies should be in ascending order.

    SCF procedures assume orbital energies are sorted from lowest
    to highest so that occupied orbitals can be selected by index.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    epsilon = h2_rhf.matrices["epsilon"]
    assert np.all(np.diff(epsilon) >= -1e-14)


def test_h2_rhf_coefficients_orthonormal(h2_rhf: RestrictedHartreeFock) -> None:
    r"""MO coefficients should satisfy C^T S C = I.

    In a non-orthogonal AO basis, MO coefficients are orthonormal
    in the S-metric.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    h2_rhf.run()
    C = h2_rhf.matrices["C"]
    S = h2_rhf.matrices["S"]
    overlap = C.T @ S @ C
    np.testing.assert_allclose(overlap, np.eye(C.shape[1]), atol=1e-10)


# ======================================================================
# SCF Run Tests — Water / STO-3G
# ======================================================================


def test_water_rhf_converges(water_rhf: RestrictedHartreeFock) -> None:
    """Water / STO-3G RHF should converge.

    Water is a standard benchmark for RHF.  Convergence with a
    minimal basis set must succeed within the default iteration limit.

    :param water_rhf: Uninitialised RHF for water/STO-3G.
    :type water_rhf: RestrictedHartreeFock
    """
    water_rhf.run()
    assert water_rhf.converged is True


def test_water_rhf_density_trace(water_rhf: RestrictedHartreeFock) -> None:
    """Tr(P S) for water should equal 10 electrons.

    Water has 10 electrons (8 from O + 2 from H's).  The density
    trace must reproduce this count.

    :param water_rhf: Uninitialised RHF for water/STO-3G.
    :type water_rhf: RestrictedHartreeFock
    """
    water_rhf.run()
    P = water_rhf.matrices["P"]
    S = water_rhf.matrices["S"]
    n_electrons = np.trace(P @ S)
    np.testing.assert_allclose(n_electrons, 10.0, atol=1e-8)


def test_water_rhf_electronic_energy_negative(
    water_rhf: RestrictedHartreeFock,
) -> None:
    """Electronic energy for water should be negative.

    :param water_rhf: Uninitialised RHF for water/STO-3G.
    :type water_rhf: RestrictedHartreeFock
    """
    water_rhf.run()
    assert water_rhf.e_electronic < 0.0
