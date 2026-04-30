"""Unit tests for compute.solvers.wavefunction.hartree_fock.hartree_fock module.

Tests cover:
- SpinPair construction (shared and independent modes)
- SpinPair property access (alpha, beta, shared, total)
- SpinPair.wrap() factory method
- HartreeFock electron-count properties
- HartreeFock Coulomb and exchange matrix construction
- HartreeFock Fock matrix construction (shared and independent density)
- HartreeFock electronic energy computation
- HartreeFock density matrix construction
- HartreeFock initial density guess
- HartreeFock result collection

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block.compute.solvers.diagonalisation import diagonalise_fock
from q_block.compute.solvers.spin_pair import SpinPair
from q_block.compute.solvers.wavefunction.hartree_fock.hartree_fock import HartreeFock
from q_block.compute.solvers.wavefunction.hartree_fock.restricted_hartree_fock import (
    RestrictedHartreeFock,
)
from q_block.compute.solvers.wavefunction.hartree_fock.unrestricted_hartree_fock import (
    UnrestrictedHartreeFock,
)
from q_block.tests.verification.utilities import h2_molecule, h2_rhf, h2_uhf

# ======================================================================
# SpinPair — Construction (shared mode)
# ======================================================================


def test_spinpair_shared_mode_construction() -> None:
    """SpinPair with alpha only should create shared mode.

    When only alpha is provided (beta=None), both channels should
    reference the same underlying array object.  This is the default
    behaviour and avoids redundant storage for closed-shell RHF.
    """
    alpha = np.array([[1.0, 0.5], [0.5, 2.0]])
    sp = SpinPair(alpha=alpha)
    assert sp.shared is True


def test_spinpair_shared_alpha_beta_same_object() -> None:
    """In shared mode, alpha and beta should be the same object.

    When beta is not supplied, both spin channels must point to the
    exact same ndarray instance (identity check with 'is').  This
    ensures no unnecessary memory duplication occurs.
    """
    alpha = np.array([[1.0, 0.0], [0.0, 1.0]])
    sp = SpinPair(alpha=alpha)
    assert sp.alpha is sp.beta


def test_spinpair_shared_tuple_length() -> None:
    """SpinPair should always have length 2 as a tuple subclass.

    Regardless of shared or independent mode, a SpinPair is a
    two-element tuple.  The length must be 2, matching the
    (alpha, beta) interface expected by downstream code.
    """
    sp = SpinPair(alpha=np.eye(5))
    assert len(sp) == 2


# ======================================================================
# SpinPair — Construction (independent mode)
# ======================================================================


def test_spinpair_independent_mode_construction() -> None:
    """SpinPair with both alpha and beta should create independent mode.

    When both spin-channel matrices are explicitly supplied, shared
    mode must be False, indicating two distinct arrays.
    """
    alpha = np.array([[1.0, 0.0], [0.0, 2.0]])
    beta = np.array([[0.5, 0.0], [0.0, 1.5]])
    sp = SpinPair(alpha=alpha, beta=beta)
    assert sp.shared is False


def test_spinpair_independent_alpha_beta_different() -> None:
    """In independent mode, alpha and beta should be different objects.

    When distinct matrices are passed for alpha and beta, the
    SpinPair must store them separately.  The content should match
    the original arrays provided at construction.
    """
    alpha = np.array([[1.0, 0.0], [0.0, 2.0]])
    beta = np.array([[0.5, 0.0], [0.0, 1.5]])
    sp = SpinPair(alpha=alpha, beta=beta)
    np.testing.assert_array_equal(sp.alpha, alpha)
    np.testing.assert_array_equal(sp.beta, beta)


# ======================================================================
# SpinPair — Property access
# ======================================================================


def test_spinpair_alpha_property() -> None:
    """The .alpha property should return the first element.

    Named access via .alpha is syntactic sugar for index 0.  Both
    must yield the same array to maintain consistency with the
    tuple interface.
    """
    alpha = np.diag([1.0, 2.0, 3.0])
    sp = SpinPair(alpha=alpha)
    np.testing.assert_array_equal(sp.alpha, sp[0])


def test_spinpair_beta_property() -> None:
    """The .beta property should return the second element.

    Named access via .beta is syntactic sugar for index 1.  For
    independent mode, this must return the explicitly provided
    beta matrix.
    """
    alpha = np.diag([1.0, 2.0])
    beta = np.diag([3.0, 4.0])
    sp = SpinPair(alpha=alpha, beta=beta)
    np.testing.assert_array_equal(sp.beta, sp[1])
    np.testing.assert_array_equal(sp.beta, beta)


def test_spinpair_total_shared() -> None:
    """In shared mode, .total should return 2 * alpha.

    When both channels are identical (shared), the total density
    is computed as 2 * alpha, avoiding a redundant element-wise
    addition.  The result must equal alpha + alpha.
    """
    alpha = np.array([[1.0, 0.5], [0.5, 2.0]])
    sp = SpinPair(alpha=alpha)
    expected = 2.0 * alpha
    np.testing.assert_allclose(sp.total, expected, atol=1e-14)


def test_spinpair_total_independent() -> None:
    """In independent mode, .total should return alpha + beta.

    When spin channels differ, the total is the element-wise sum
    of both matrices.  This must match a direct numpy addition.
    """
    alpha = np.array([[1.0, 0.0], [0.0, 2.0]])
    beta = np.array([[0.5, 0.1], [0.1, 1.5]])
    sp = SpinPair(alpha=alpha, beta=beta)
    expected = alpha + beta
    np.testing.assert_allclose(sp.total, expected, atol=1e-14)


# ======================================================================
# SpinPair — wrap factory method
# ======================================================================


def test_spinpair_wrap_returns_same_if_already_spinpair() -> None:
    """SpinPair.wrap should return the same object for a SpinPair input.

    If the argument is already a SpinPair, wrap() must return it
    unchanged (identity check), avoiding unnecessary re-wrapping.
    """
    sp = SpinPair(alpha=np.eye(2))
    wrapped = SpinPair.wrap(obj=sp)
    assert wrapped is sp


def test_spinpair_wrap_converts_plain_tuple() -> None:
    """SpinPair.wrap should convert a plain tuple into a SpinPair.

    A plain (array, array) tuple should be wrapped into a non-shared
    SpinPair with the first element as alpha and the second as beta.
    """
    alpha = np.eye(2)
    beta = np.eye(2) * 0.5
    plain = (alpha, beta)
    wrapped = SpinPair.wrap(obj=plain)
    assert isinstance(wrapped, SpinPair)
    np.testing.assert_array_equal(wrapped.alpha, alpha)
    np.testing.assert_array_equal(wrapped.beta, beta)


def test_spinpair_wrap_result_is_not_shared() -> None:
    """SpinPair.wrap from a plain tuple should produce non-shared mode.

    The wrap() method always creates independent mode from a plain
    tuple because it cannot know whether the two arrays are meant
    to be the same.
    """
    alpha = np.eye(2)
    wrapped = SpinPair.wrap(obj=(alpha, alpha))
    assert wrapped.shared is False


# ======================================================================
# HartreeFock — Electron-count properties
# ======================================================================


def test_n_alpha_property(h2_rhf: RestrictedHartreeFock) -> None:
    """n_alpha should return the number of alpha electrons.

    For H2 with RHF (2 electrons, n_occ=1), n_alpha must be 1
    because each spatial orbital holds one alpha and one beta.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.n_alpha == 1


def test_n_beta_property(h2_rhf: RestrictedHartreeFock) -> None:
    """n_beta should return the number of beta electrons.

    For H2 with RHF, n_beta must equal n_alpha (closed-shell).

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.n_beta == 1


def test_n_electrons_property(h2_rhf: RestrictedHartreeFock) -> None:
    """n_electrons should return the total electron count.

    The total must be the sum of alpha and beta electrons.  For H2
    this is 2.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    assert h2_rhf.n_electrons == 2


def test_n_electrons_uhf(h2_uhf: UnrestrictedHartreeFock) -> None:
    """UHF n_electrons should sum alpha and beta counts.

    For H2 in UHF with n_alpha=1, n_beta=1 the total must be 2.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    assert h2_uhf.n_electrons == 2


# ======================================================================
# HartreeFock — Coulomb and Exchange matrices
# ======================================================================


def test_build_coulomb_shape(h2_rhf: RestrictedHartreeFock) -> None:
    """Coulomb matrix should be n_basis × n_basis.

    The Coulomb matrix J is built from a density matrix and the
    ERI tensor.  Its dimensions must match the number of basis
    functions.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    P = np.eye(n)
    J = h2_rhf._build_coulomb(P=P)
    assert J.shape == (n, n)


def test_build_coulomb_symmetry(h2_rhf: RestrictedHartreeFock) -> None:
    """Coulomb matrix should be symmetric.

    The Coulomb integral J_mn = sum P_ls (mn|ls) is symmetric in
    m and n because the ERI tensor has the permutational symmetry
    (mn|ls) = (nm|ls).

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    rng = np.random.default_rng(42)
    P = rng.random((n, n))
    P = P + P.T
    J = h2_rhf._build_coulomb(P=P)
    np.testing.assert_allclose(J, J.T, atol=1e-12)


def test_build_exchange_shape(h2_rhf: RestrictedHartreeFock) -> None:
    """Exchange matrix should be n_basis × n_basis.

    The exchange matrix K is built analogously to J but with
    permuted indices.  Its dimensions must also match the basis size.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    P = np.eye(n)
    K = h2_rhf._build_exchange(P=P)
    assert K.shape == (n, n)


def test_build_exchange_symmetry(h2_rhf: RestrictedHartreeFock) -> None:
    """Exchange matrix should be symmetric for a symmetric density.

    When the input density P is symmetric, K inherits symmetry
    from the permutational symmetry of the ERI tensor.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    rng = np.random.default_rng(43)
    P = rng.random((n, n))
    P = P + P.T
    K = h2_rhf._build_exchange(P=P)
    np.testing.assert_allclose(K, K.T, atol=1e-12)


# ======================================================================
# HartreeFock — Fock matrix construction
# ======================================================================


def test_build_fock_shared_density(h2_rhf: RestrictedHartreeFock) -> None:
    """Fock build with shared density should return a shared SpinPair.

    When the input density is shared (RHF case), only one exchange
    matrix is computed and the returned SpinPair is also shared.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    P = SpinPair(alpha=np.eye(n) * 0.5)
    fock = h2_rhf._build_fock(density=P)
    assert isinstance(fock, SpinPair)
    assert fock.shared is True


def test_build_fock_independent_density(h2_uhf: UnrestrictedHartreeFock) -> None:
    """Fock build with independent density should return independent SpinPair.

    When alpha and beta densities differ (UHF case), the returned
    Fock pair must be independent with different alpha and beta
    Fock matrices.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    n = h2_uhf.n_basis
    Pa = np.eye(n) * 0.6
    Pb = np.eye(n) * 0.4
    density = SpinPair(alpha=Pa, beta=Pb)
    fock = h2_uhf._build_fock(density=density)
    assert isinstance(fock, SpinPair)
    assert fock.shared is False


def test_build_fock_shape(h2_rhf: RestrictedHartreeFock) -> None:
    """Fock matrices should be n_basis × n_basis.

    Both alpha and beta Fock matrices from _build_fock must have
    the correct square shape matching the basis set size.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    P = SpinPair(alpha=np.eye(n) * 0.5)
    fock = h2_rhf._build_fock(density=P)
    assert fock.alpha.shape == (n, n)
    assert fock.beta.shape == (n, n)


def test_build_fock_symmetry(h2_rhf: RestrictedHartreeFock) -> None:
    """Fock matrix should be symmetric for symmetric density.

    The Fock matrix F = H + J - K inherits symmetry from H, J, and K,
    all of which are symmetric for a symmetric density.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    rng = np.random.default_rng(44)
    P_arr = rng.random((n, n))
    P_arr = P_arr + P_arr.T
    P = SpinPair(alpha=P_arr)
    fock = h2_rhf._build_fock(density=P)
    np.testing.assert_allclose(fock.alpha, fock.alpha.T, atol=1e-12)


# ======================================================================
# HartreeFock — Electronic energy computation
# ======================================================================


def test_compute_electronic_energy_negative(h2_rhf: RestrictedHartreeFock) -> None:
    """Electronic energy from a reasonable density should be negative.

    For a bound molecular system the electronic energy (including
    electron-electron repulsion but excluding nuclear repulsion)
    must be negative.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._initial_density(C=C)
    fock = h2_rhf._build_fock(density=density)
    e_elec = h2_rhf._compute_electronic_energy(density=density, fock=fock)
    assert e_elec < 0.0


def test_compute_electronic_energy_is_scalar(h2_rhf: RestrictedHartreeFock) -> None:
    """Electronic energy should return a Python float.

    The energy must be a scalar value, not an array.  This ensures
    that downstream convergence checks work correctly.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._initial_density(C=C)
    fock = h2_rhf._build_fock(density=density)
    e_elec = h2_rhf._compute_electronic_energy(density=density, fock=fock)
    assert isinstance(e_elec, (float, np.floating))


# ======================================================================
# HartreeFock — Density matrix construction
# ======================================================================


def test_build_density_shared_for_rhf(h2_rhf: RestrictedHartreeFock) -> None:
    """_build_density for RHF should return a shared SpinPair.

    In closed-shell RHF, alpha and beta densities are identical.
    The returned SpinPair must be in shared mode.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    C_pair = (C, C)
    density = h2_rhf._build_density(C=C_pair)
    assert isinstance(density, SpinPair)
    assert density.shared is True


def test_build_density_independent_for_uhf(h2_uhf: UnrestrictedHartreeFock) -> None:
    """_build_density for UHF should return an independent SpinPair.

    In unrestricted HF, alpha and beta densities generally differ.
    The returned SpinPair must be in independent mode.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_uhf.H, X=h2_uhf.X)
    C_pair = (C, C)
    density = h2_uhf._build_density(C=C_pair)
    assert isinstance(density, SpinPair)
    assert density.shared is False


def test_build_density_shape(h2_rhf: RestrictedHartreeFock) -> None:
    """Density matrices should be n_basis × n_basis.

    Density matrices are projectors in the AO basis and must
    be square with dimension equal to the number of basis functions.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._build_density(C=(C, C))
    assert density.alpha.shape == (n, n)


def test_build_density_symmetric(h2_rhf: RestrictedHartreeFock) -> None:
    """Density matrix should be symmetric.

    Because P = C_occ @ C_occ^T, the density matrix is always
    symmetric.  Asymmetry would indicate incorrect slicing of
    occupied orbitals.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._build_density(C=(C, C))
    np.testing.assert_allclose(density.alpha, density.alpha.T, atol=1e-14)


# ======================================================================
# HartreeFock — Initial density guess
# ======================================================================


def test_initial_density_returns_spinpair(h2_rhf: RestrictedHartreeFock) -> None:
    """_initial_density should return a SpinPair.

    The initial guess density constructed from core-Hamiltonian MO
    coefficients must be returned as a SpinPair for consistency
    with the SCF loop.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._initial_density(C=C)
    assert isinstance(density, SpinPair)


def test_initial_density_shared_for_rhf(h2_rhf: RestrictedHartreeFock) -> None:
    """_initial_density for RHF should be shared.

    The initial density for closed-shell RHF uses the same MO
    coefficients for both channels, so the SpinPair must be shared.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_rhf.H, X=h2_rhf.X)
    density = h2_rhf._initial_density(C=C)
    assert density.shared is True


def test_initial_density_independent_for_uhf(
    h2_uhf: UnrestrictedHartreeFock,
) -> None:
    """_initial_density for UHF should be independent.

    For UHF the initial density is built with different occupation
    numbers for alpha and beta, producing an independent SpinPair.

    :param h2_uhf: Uninitialised UHF for H2/STO-3G.
    :type h2_uhf: UnrestrictedHartreeFock
    """
    C, _ = diagonalise_fock(fock=h2_uhf.H, X=h2_uhf.X)
    density = h2_uhf._initial_density(C=C)
    assert isinstance(density, SpinPair)
    assert density.shared is False


# ======================================================================
# HartreeFock — collect_results
# ======================================================================


def test_collect_results_sets_converged(h2_rhf: RestrictedHartreeFock) -> None:
    """_collect_results should set the converged attribute.

    After calling _collect_results, the converged attribute must
    reflect the value passed to it (True or False).

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    dummy = np.zeros((n, n))
    dummy_eps = np.zeros(n)
    sp = SpinPair(alpha=dummy)
    sp_eps = SpinPair(alpha=dummy_eps)

    h2_rhf._collect_results(
        converged=True,
        n_iterations=5,
        e_electronic=-1.0,
        fock=sp,
        density=sp,
        C=sp,
        epsilon=sp_eps,
    )
    assert h2_rhf.converged is True


def test_collect_results_sets_e_total(h2_rhf: RestrictedHartreeFock) -> None:
    """_collect_results should compute e_total = e_electronic + e_nuclear.

    The total energy combines the electronic energy with the
    precomputed nuclear repulsion energy.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    dummy = np.zeros((n, n))
    dummy_eps = np.zeros(n)
    sp = SpinPair(alpha=dummy)
    sp_eps = SpinPair(alpha=dummy_eps)

    h2_rhf._collect_results(
        converged=True,
        n_iterations=5,
        e_electronic=-2.0,
        fock=sp,
        density=sp,
        C=sp,
        epsilon=sp_eps,
    )
    expected = -2.0 + h2_rhf.e_nuclear
    np.testing.assert_allclose(h2_rhf.e_total, expected, atol=1e-14)


def test_collect_results_wraps_plain_tuples(h2_rhf: RestrictedHartreeFock) -> None:
    """_collect_results should accept plain tuples and wrap to SpinPair.

    The method uses SpinPair.wrap() internally so that downstream
    code (_store_matrices) always receives SpinPair instances.

    :param h2_rhf: Uninitialised RHF for H2/STO-3G.
    :type h2_rhf: RestrictedHartreeFock
    """
    n = h2_rhf.n_basis
    dummy = np.zeros((n, n))
    dummy_eps = np.zeros(n)
    plain_tuple = (dummy, dummy)
    plain_eps = (dummy_eps, dummy_eps)

    # Should not raise when plain tuples are passed.
    h2_rhf._collect_results(
        converged=True,
        n_iterations=3,
        e_electronic=-1.5,
        fock=plain_tuple,
        density=plain_tuple,
        C=plain_tuple,
        epsilon=plain_eps,
    )
    assert h2_rhf.converged is True
