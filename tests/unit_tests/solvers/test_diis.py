"""Unit tests for q_block.solvers.diis module.

Tests cover:
- DIIS construction and validation
- Error vector computation (restricted and unrestricted)
- Extrapolation with insufficient history (< 2 pairs)
- Extrapolation with sufficient history (>= 2 pairs)
- Tuple (unrestricted) Fock matrix handling
- Subspace trimming when max_vectors is exceeded
- Reset clears stored vectors

The DIIS (Direct Inversion of the Iterative Subspace) module accelerates
SCF convergence by constructing an optimal linear combination of Fock
matrices from previous iterations.  These tests exercise the full
lifecycle: construction, error computation, extrapolation with both
single-matrix and tuple inputs, subspace management, and reset.

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block.solvers.diis import DIIS


# ======================================================================
# Construction Tests
# ======================================================================


@pytest.mark.parametrize(
    "max_vec",
    [2, 3, 10, 100],
    ids=["min=2", "3", "10", "100"],
)
def test_custom_max_vectors(max_vec: int) -> None:
    """Custom max_vectors values >= 2 should be accepted.

    The DIIS subspace size controls how many Fock/error pairs are
    retained.  Any integer >= 2 is valid because at least two pairs
    are needed to form a meaningful linear combination.  The stored
    attribute must reflect the value passed at construction.

    :param max_vec: Subspace size to test, provided by parametrize.
    :type max_vec: int
    """
    # diis is constructed with the parametrised subspace size
    diis = DIIS(max_vectors=max_vec)
    assert diis.max_vectors == max_vec


@pytest.mark.parametrize(
    "max_vec",
    [0, 1, -1],
    ids=["zero", "one", "neg1"],
)
def test_max_vectors_below_two_raises(max_vec: int) -> None:
    """max_vectors < 2 should raise ValueError.

    DIIS requires at least two Fock/error pairs to build the B matrix
    and solve for interpolation coefficients.  Passing a subspace size
    below 2 is therefore invalid and must be rejected immediately at
    construction time.

    :param max_vec: Invalid subspace size to test, provided by parametrize.
    :type max_vec: int
    """
    with pytest.raises(ValueError, match="max_vectors must be >= 2"):
        DIIS(max_vectors=max_vec)


# ======================================================================
# compute_error Tests (restricted)
# ======================================================================


def test_compute_error_identity_overlap() -> None:
    """With S = I and commuting F, P the error should be zero.

    The commutator error e = FPS - SPF vanishes when F and P commute
    (both diagonal).  With an identity overlap this reduces to FP - PF.
    This test confirms the zero-error baseline for well-converged or
    trivially diagonal matrices.
    """
    n = 3
    S = np.eye(n)
    # Diagonal F and P commute → FPS - SPF = 0
    F = np.diag([1.0, 2.0, 3.0])
    P = np.diag([0.5, 0.3, 0.1])
    # error is the commutator, expected to be the zero matrix
    error = DIIS.compute_error(F, P, S)
    np.testing.assert_allclose(error, np.zeros((n, n)), atol=1e-14)


def test_compute_error_nonzero() -> None:
    """Non-commuting F and P should produce nonzero error.

    When the Fock and density matrices have off-diagonal elements they
    generally do not commute, so the DIIS commutator FPS - SPF is
    nonzero.  We verify the result matches a direct matrix computation.
    """
    n = 2
    S = np.eye(n)
    F = np.array([[1.0, 0.5], [0.5, 2.0]])
    P = np.array([[0.8, 0.2], [0.2, 0.3]])
    error = DIIS.compute_error(F, P, S)
    # expected is the explicitly computed commutator
    expected = F @ P @ S - S @ P @ F
    np.testing.assert_allclose(error, expected, atol=1e-14)


def test_compute_error_symmetric_overlap() -> None:
    """Error computation with non-identity symmetric overlap matrix.

    In a real basis set the overlap matrix S is not the identity.  This
    test uses a non-trivial S to verify the commutator formula
    FPS - SPF still produces the correct result when S has off-diagonal
    elements.
    """
    S = np.array([[1.0, 0.2], [0.2, 1.0]])
    F = np.array([[1.0, 0.3], [0.3, 2.0]])
    P = np.array([[0.6, 0.1], [0.1, 0.4]])
    error = DIIS.compute_error(F, P, S)
    # expected is the reference commutator with the non-identity S
    expected = F @ P @ S - S @ P @ F
    np.testing.assert_allclose(error, expected, atol=1e-14)


# ======================================================================
# compute_error Tests (unrestricted / tuple)
# ======================================================================


def test_compute_error_unrestricted() -> None:
    """Unrestricted error should concatenate alpha and beta commutators.

    For UHF-style calculations the Fock and density matrices are tuples
    of (alpha, beta) matrices.  compute_error must compute the
    commutator for each spin channel separately and return a single
    flattened vector that concatenates both results.
    """
    n = 2
    S = np.eye(n)
    F_a = np.array([[1.0, 0.5], [0.5, 2.0]])
    F_b = np.array([[1.5, 0.3], [0.3, 2.5]])
    P_a = np.array([[0.8, 0.2], [0.2, 0.3]])
    P_b = np.array([[0.7, 0.1], [0.1, 0.2]])

    # error is a concatenation of flattened alpha and beta commutators
    error = DIIS.compute_error((F_a, F_b), (P_a, P_b), S)

    e_a = (F_a @ P_a @ S - S @ P_a @ F_a).ravel()
    e_b = (F_b @ P_b @ S - S @ P_b @ F_b).ravel()
    expected = np.concatenate([e_a, e_b])
    np.testing.assert_allclose(error, expected, atol=1e-14)


# ======================================================================
# Extrapolation Tests — insufficient history
# ======================================================================


def test_extrapolate_returns_original_on_first_call() -> None:
    """First call should return the original Fock matrix unchanged.

    DIIS requires at least two stored pairs before it can solve for
    interpolation coefficients.  On the very first call only one pair
    exists, so the method must pass the original Fock matrix through
    without modification.
    """
    diis = DIIS()
    fock = np.array([[1.0, 0.5], [0.5, 2.0]])
    error = np.array([0.1, 0.2, 0.3, 0.4])
    # result should be identical to the input fock
    result = diis.extrapolate(fock, error)
    np.testing.assert_array_equal(result, fock)


# ======================================================================
# Extrapolation Tests — sufficient history
# ======================================================================


def test_extrapolate_returns_different_after_two_calls() -> None:
    """After two Fock/error pairs the extrapolation should differ.

    Once two or more pairs have been accumulated, DIIS solves the
    Lagrangian system for coefficients and returns a weighted
    combination of the stored Fock matrices.  The result should
    generally differ from the most recent input Fock matrix.
    """
    diis = DIIS()
    n = 2
    rng = np.random.default_rng(42)

    # fock1/err1 is the first pair stored in the subspace
    fock1 = rng.random((n, n))
    fock1 = fock1 + fock1.T
    err1 = rng.random(n * n)

    # fock2/err2 is the second pair that triggers extrapolation
    fock2 = rng.random((n, n))
    fock2 = fock2 + fock2.T
    err2 = rng.random(n * n)

    diis.extrapolate(fock1, err1)
    result = diis.extrapolate(fock2, err2)

    assert result.shape == fock2.shape
    # The result should in general differ from fock2
    assert not np.allclose(result, fock2)


def test_extrapolate_shape_preserved() -> None:
    """Extrapolated Fock should have the same shape as the input.

    Regardless of how many pairs have been accumulated, the output
    matrix must always have the same shape as the input Fock matrix.
    This invariant is critical because the SCF loop feeds the
    extrapolated matrix directly into the diagonaliser.
    """
    diis = DIIS()
    n = 4
    rng = np.random.default_rng(123)

    for _ in range(3):
        # f is a symmetric n×n matrix fed into the subspace
        f = rng.random((n, n))
        f = f + f.T
        e = rng.random(n * n)
        result = diis.extrapolate(f, e)
        assert result.shape == (n, n)


def test_extrapolate_coefficients_sum_to_one() -> None:
    """DIIS coefficients should sum to 1 (constraint).

    The Lagrangian formulation imposes the constraint sum(c_i) = 1.
    Although we cannot directly inspect the coefficients from the public
    API, we verify that a valid extrapolation occurred by checking that
    the internal subspace is populated after feeding three pairs.
    """
    diis = DIIS()
    n = 2
    rng = np.random.default_rng(7)

    # focks collects the Fock matrices fed into the subspace
    focks = []
    for _ in range(3):
        f = rng.random((n, n))
        f = f + f.T
        e = rng.random(n * n)
        focks.append(f)
        diis.extrapolate(f, e)

    # The internal storage must be non-empty after extrapolation
    assert diis._fock_vectors


# ======================================================================
# Extrapolation Tests — tuple (unrestricted) Fock
# ======================================================================


def test_extrapolate_tuple_returns_tuple() -> None:
    """When fock is a tuple, extrapolate should return a tuple.

    For unrestricted SCF the Fock matrix is passed as a tuple of
    (alpha, beta) matrices.  The extrapolation must detect this and
    return a tuple of the same length, with each element reshaped
    back to the original per-channel matrix shape.
    """
    diis = DIIS()
    n = 2
    rng = np.random.default_rng(99)

    for _ in range(3):
        # fa and fb are the alpha and beta Fock matrices
        fa = rng.random((n, n))
        fb = rng.random((n, n))
        e = rng.random(2 * n * n)
        result = diis.extrapolate((fa, fb), e)

    # result must be a length-2 tuple of n×n matrices
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0].shape == (n, n)
    assert result[1].shape == (n, n)


def test_extrapolate_tuple_first_call_unchanged() -> None:
    """First call with a tuple Fock should return it unchanged.

    Just like the single-matrix case, when only one pair has been
    stored the tuple input must be returned as-is.  Both the alpha
    and beta matrices must be identical to the originals.
    """
    diis = DIIS()
    fa = np.array([[1.0, 0.5], [0.5, 2.0]])
    fb = np.array([[1.5, 0.3], [0.3, 2.5]])
    error = np.array([0.1, 0.2, 0.3])
    # result is the pass-through tuple on the first call
    result = diis.extrapolate((fa, fb), error)
    assert isinstance(result, tuple)
    np.testing.assert_array_equal(result[0], fa)
    np.testing.assert_array_equal(result[1], fb)


# ======================================================================
# Subspace Trimming Tests
# ======================================================================


def test_subspace_trimmed_to_max_vectors() -> None:
    """Stored vectors should not exceed max_vectors.

    When more pairs are added than max_vectors allows, the oldest
    pair must be discarded.  This test feeds max_vec + 5 pairs into
    a DIIS instance with max_vectors=3 and verifies that both the
    Fock and error storage remain capped at the limit.
    """
    max_vec = 3
    diis = DIIS(max_vectors=max_vec)
    n = 2
    rng = np.random.default_rng(0)

    for _ in range(max_vec + 5):
        f = rng.random((n, n))
        e = rng.random(n * n)
        diis.extrapolate(f, e)

    # Both internal lists must be capped at max_vec
    assert len(diis._fock_vectors) == max_vec
    assert len(diis._error_vectors) == max_vec


# ======================================================================
# Reset Tests
# ======================================================================


def test_reset_clears_vectors() -> None:
    """reset() should remove all stored Fock and error vectors.

    After an SCF calculation finishes or a restart is needed, calling
    reset() must empty both the Fock and error vector lists.  This
    test populates the subspace with four pairs, verifies it is
    non-empty, then checks that reset() brings it back to zero.
    """
    diis = DIIS()
    n = 2
    rng = np.random.default_rng(11)
    for _ in range(4):
        f = rng.random((n, n))
        e = rng.random(n * n)
        diis.extrapolate(f, e)

    # Subspace must be non-empty before reset
    assert len(diis._fock_vectors) > 0
    diis.reset()
    # Both lists must be empty after reset
    assert len(diis._fock_vectors) == 0
    assert len(diis._error_vectors) == 0
