"""Unit tests for compute.solvers.diagonalisation module.

Tests cover:
- diagonalise_single: eigenvalue problem with identity and non-identity X
- diagonalise_fock: single matrix and tuple (unrestricted) dispatching
- Eigenvalue ordering (ascending)
- Back-transformation correctness
- Orthogonality of MO coefficients in the S-metric

The diagonalisation module solves the Roothaan–Hall generalised eigenvalue
problem F C = S C epsilon by transforming to an orthogonal basis via
X = S^{-1/2}.  These tests verify correctness of that transformation,
eigenvalue ordering, and proper handling of both restricted (single matrix)
and unrestricted (tuple of matrices) Fock inputs.

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest
from scipy.linalg import eigh, fractional_matrix_power

from compute.solvers.diagonalisation import diagonalise_fock, diagonalise_single
from tests.unit_tests.utilities import make_overlap, make_symmetric

# ======================================================================
# diagonalise_single Tests — identity basis
# ======================================================================


def test_single_identity_X_eigenvalues() -> None:
    """With X = I the eigenvalues should match scipy.linalg.eigh.

    When the orthogonalisation matrix is the identity the generalised
    eigenvalue problem reduces to a standard one.  The resulting orbital
    energies must therefore match the reference values returned by
    scipy.linalg.eigh applied directly to the Fock matrix.
    """
    n = 4
    # F is a random symmetric Fock-like matrix
    F = make_symmetric(n, seed=10)
    X = np.eye(n)

    C, epsilon = diagonalise_single(F, X)

    # eps_ref are the reference eigenvalues from scipy
    eps_ref, _ = eigh(F)
    np.testing.assert_allclose(epsilon, eps_ref, atol=1e-12)


def test_single_identity_X_eigenvectors() -> None:
    """With X = I the eigenvector equation F C = C diag(epsilon) must hold.

    This verifies that the returned MO coefficients satisfy the standard
    eigenvalue equation.  The product F @ C must equal C @ diag(epsilon)
    to within numerical precision, confirming the back-transformation
    from the orthogonal basis is correct.
    """
    n = 4
    # F is a random symmetric Fock-like matrix
    F = make_symmetric(n, seed=20)
    X = np.eye(n)

    C, epsilon = diagonalise_single(F, X)

    # Verify the eigenvalue equation F C = C diag(epsilon)
    np.testing.assert_allclose(F @ C, C @ np.diag(epsilon), atol=1e-12)


def test_single_eigenvalues_ascending() -> None:
    """Eigenvalues should be returned in ascending order.

    SCF procedures assume orbital energies are sorted from lowest to
    highest so that occupied orbitals can be selected by index.  This
    test checks that consecutive differences are non-negative, ensuring
    the sort invariant holds.
    """
    n = 5
    # F is a 5x5 symmetric matrix producing 5 eigenvalues
    F = make_symmetric(n, seed=30)
    X = np.eye(n)

    _, epsilon = diagonalise_single(F, X)
    # diff checks that each eigenvalue is >= the previous one
    assert np.all(np.diff(epsilon) >= -1e-14)


# ======================================================================
# diagonalise_single Tests — non-trivial X
# ======================================================================


def test_single_nontrivial_X_solves_generalised_problem() -> None:
    r"""With X = S^{-1/2} the result should solve F C = S C epsilon.

    This is the core correctness check for the Roothaan–Hall procedure.
    A non-trivial overlap matrix S is generated, from which the
    orthogonalisation matrix X = S^{-1/2} is computed.  After
    diagonalisation, the generalised eigenvalue relation
    F C = S C diag(epsilon) must be satisfied.
    """
    n = 3
    # S is a positive-definite overlap matrix
    S = make_overlap(n, seed=40)
    X = np.real(fractional_matrix_power(S, -0.5))
    F = make_symmetric(n, seed=41)

    C, epsilon = diagonalise_single(F, X)

    # Verify the generalised eigenvalue equation F C = S C diag(epsilon)
    np.testing.assert_allclose(F @ C, S @ C @ np.diag(epsilon), atol=1e-10)


def test_single_nontrivial_X_orthonormality() -> None:
    r"""MO coefficients should be orthonormal in the S-metric: C^T S C = I.

    In a non-orthogonal AO basis the MO coefficients are not simply
    orthogonal in the Euclidean sense; instead they satisfy
    C^T S C = I.  This property is essential for the density matrix
    construction and must be preserved by the back-transformation.
    """
    n = 3
    # S is the non-trivial overlap, X is the orthogonalisation matrix
    S = make_overlap(n, seed=50)
    X = np.real(fractional_matrix_power(S, -0.5))
    F = make_symmetric(n, seed=51)

    C, _ = diagonalise_single(F, X)

    # overlap should be the identity in the S-metric
    overlap = C.T @ S @ C
    np.testing.assert_allclose(overlap, np.eye(n), atol=1e-10)


# ======================================================================
# diagonalise_fock Tests — single matrix
# ======================================================================


def test_fock_single_matches_diagonalise_single() -> None:
    """diagonalise_fock with a single matrix should match diagonalise_single.

    The wrapper function diagonalise_fock dispatches to diagonalise_single
    when given a plain array.  Both code paths must produce identical
    eigenvalues and eigenvectors (up to sign) for the same input.
    """
    n = 3
    # F is the same input fed to both functions for comparison
    F = make_symmetric(n, seed=60)
    X = np.eye(n)

    C_fock, eps_fock = diagonalise_fock(F, X)
    C_single, eps_single = diagonalise_single(F, X)

    np.testing.assert_allclose(eps_fock, eps_single, atol=1e-14)
    # Compare absolute values because eigenvectors may differ by sign
    np.testing.assert_allclose(np.abs(C_fock), np.abs(C_single), atol=1e-14)


def test_fock_single_returns_arrays() -> None:
    """diagonalise_fock with a single matrix should return plain arrays.

    When the input is a single ndarray (restricted case), the outputs
    must also be plain ndarrays, not tuples.  This ensures downstream
    code can use matrix operations directly without unpacking.
    """
    n = 3
    # F is a plain ndarray, not a tuple
    F = make_symmetric(n, seed=70)
    X = np.eye(n)

    C, epsilon = diagonalise_fock(F, X)

    # Both outputs must be plain ndarrays
    assert isinstance(C, np.ndarray)
    assert isinstance(epsilon, np.ndarray)
    assert not isinstance(C, tuple)
    assert not isinstance(epsilon, tuple)


# ======================================================================
# diagonalise_fock Tests — tuple (unrestricted)
# ======================================================================


def test_fock_tuple_returns_tuples() -> None:
    """diagonalise_fock with a tuple should return tuples of arrays.

    For unrestricted calculations the Fock matrix is a tuple of
    spin-channel matrices (alpha, beta).  The function must detect
    the tuple input and return matching tuples of MO coefficients
    and orbital energies, one per spin channel.
    """
    n = 3
    # F_a and F_b represent alpha and beta Fock matrices
    F_a = make_symmetric(n, seed=80)
    F_b = make_symmetric(n, seed=81)
    X = np.eye(n)

    C, epsilon = diagonalise_fock((F_a, F_b), X)

    # Both outputs must be tuples with two elements
    assert isinstance(C, tuple)
    assert isinstance(epsilon, tuple)
    assert len(C) == 2
    assert len(epsilon) == 2


def test_fock_tuple_each_channel_correct() -> None:
    """Each spin channel should be diagonalised independently.

    The tuple dispatch must diagonalise each Fock matrix in isolation.
    We verify this by comparing the tuple results element-by-element
    against separate diagonalise_single calls for each spin channel.
    Eigenvectors are compared by absolute value to account for
    arbitrary sign conventions.
    """
    n = 3
    X = np.eye(n)
    # F_a and F_b are different Fock matrices for alpha and beta spins
    F_a = make_symmetric(n, seed=90)
    F_b = make_symmetric(n, seed=91)

    C_tuple, eps_tuple = diagonalise_fock((F_a, F_b), X)
    # Reference results from individual single calls
    C_a_ref, eps_a_ref = diagonalise_single(F_a, X)
    C_b_ref, eps_b_ref = diagonalise_single(F_b, X)

    np.testing.assert_allclose(eps_tuple[0], eps_a_ref, atol=1e-14)
    np.testing.assert_allclose(eps_tuple[1], eps_b_ref, atol=1e-14)
    np.testing.assert_allclose(np.abs(C_tuple[0]), np.abs(C_a_ref), atol=1e-14)
    np.testing.assert_allclose(np.abs(C_tuple[1]), np.abs(C_b_ref), atol=1e-14)


# ======================================================================
# Shape Tests
# ======================================================================


@pytest.mark.parametrize(
    "n",
    [2, 3, 5, 8],
    ids=["n=2", "n=3", "n=5", "n=8"],
)
def test_output_shapes(n: int) -> None:
    """Output shapes should match the input basis size.

    The MO coefficient matrix C must be (n, n) and the orbital
    energy vector epsilon must be (n,), where n is the number of
    basis functions.  This invariant must hold regardless of the
    particular matrix dimension.

    :param n: Basis-set size to test, provided by parametrize.
    :type n: int
    """
    # F is an n×n symmetric matrix parametrised over several sizes
    F = make_symmetric(n, seed=100 + n)
    X = np.eye(n)

    C, epsilon = diagonalise_single(F, X)
    assert C.shape == (n, n)
    assert epsilon.shape == (n,)
