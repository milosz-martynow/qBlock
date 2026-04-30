"""Unit tests for compute.solvers.calculation_error_metric module.

Tests cover:
- RMS metric computation for various error arrays
- Max-abs metric computation for various error arrays
- Default metric selection
- Validation of unsupported metric names

The CalculationErrorMetric class reduces an error vector (or matrix) to a
scalar convergence measure used in the SCF loop.  Two reduction strategies
are supported: root-mean-square ("rms") and maximum absolute value
("max_abs").  These tests verify both static methods and the dispatch
logic of the compute() instance method.

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block.compute.solvers.calculation_error_metric import CalculationErrorMetric

# ======================================================================
# Construction Tests
# ======================================================================


def test_default_metric_is_rms() -> None:
    """Default metric should be 'rms'.

    When no metric name is provided, the constructor must default to
    root-mean-square.  This is the most commonly used convergence
    criterion in Hartree-Fock SCF procedures and serves as a sensible
    default for most calculations.
    """
    # metric is the instance created with default arguments
    metric = CalculationErrorMetric()
    assert metric.metric == "rms"


@pytest.mark.parametrize(
    "name",
    ["rms", "max_abs"],
    ids=["rms", "max_abs"],
)
def test_supported_metrics_accepted(name: str) -> None:
    """All supported metric names should be accepted without error.

    The class supports exactly two metric names: 'rms' and 'max_abs'.
    Both must be accepted at construction time and stored in the
    metric attribute.  Any other string should be rejected (tested
    separately in the unsupported-metric test).

    :param name: Metric name to test, provided by parametrize.
    :type name: str
    """
    # metric is the instance constructed with the parametrised name
    metric = CalculationErrorMetric(metric=name)
    assert metric.metric == name


@pytest.mark.parametrize(
    "name",
    ["RMS", "MAX_ABS", "l2", "frobenius", ""],
    ids=["RMS-upper", "MAX_ABS-upper", "l2", "frobenius", "empty"],
)
def test_unsupported_metric_raises(name: str) -> None:
    """Unsupported metric names should raise ValueError.

    Metric names are case-sensitive and only 'rms' and 'max_abs' are
    valid.  Upper-case variants, common alternatives like 'l2' or
    'frobenius', and the empty string must all raise a ValueError
    with a descriptive message listing the supported options.

    :param name: Invalid metric name to test, provided by parametrize.
    :type name: str
    """
    with pytest.raises(ValueError, match="Unknown metric"):
        CalculationErrorMetric(metric=name)


# ======================================================================
# RMS Computation Tests
# ======================================================================


@pytest.mark.parametrize(
    "error, expected_rms",
    [
        (np.zeros(5), 0.0),
        (np.ones(4), 1.0),
        (np.array([3.0, 4.0]), np.sqrt(25.0 / 2)),
        (np.array([1.0, -1.0, 1.0, -1.0]), 1.0),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), np.sqrt(30.0 / 4)),
    ],
    ids=["zeros", "ones", "3-4-vec", "alternating-signs", "2x2-matrix"],
)
def test_rms_static(error: np.ndarray, expected_rms: float) -> None:
    """Verify CalculationErrorMetric.rms returns correct values.

    The RMS metric flattens the error array, computes the dot product
    with itself, divides by the number of elements, and takes the
    square root.  We test with zero arrays (trivial case), uniform
    arrays, small vectors, signed inputs, and 2D matrices to confirm
    the flattening logic works correctly.

    :param error: Input error array (vector or matrix), provided by parametrize.
    :type error: np.ndarray
    :param expected_rms: Expected RMS value for the given error, provided by parametrize.
    :type expected_rms: float
    """
    # result is the scalar RMS value computed by the static method
    result = CalculationErrorMetric.rms(error)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected_rms, atol=1e-14)


@pytest.mark.parametrize(
    "error, expected_rms",
    [
        (np.zeros(5), 0.0),
        (np.ones(4), 1.0),
        (np.array([3.0, 4.0]), np.sqrt(25.0 / 2)),
    ],
    ids=["zeros", "ones", "3-4-vec"],
)
def test_compute_dispatches_to_rms(error: np.ndarray, expected_rms: float) -> None:
    """compute() with 'rms' metric should dispatch to rms().

    The compute() instance method selects the reduction strategy at
    runtime based on the metric attribute.  When set to 'rms' it must
    produce the same result as calling the static rms() method
    directly, confirming the dispatch logic is correct.

    :param error: Input error array, provided by parametrize.
    :type error: np.ndarray
    :param expected_rms: Expected RMS value, provided by parametrize.
    :type expected_rms: float
    """
    metric = CalculationErrorMetric(metric="rms")
    # result should equal the static rms() output
    result = metric.compute(error)
    np.testing.assert_allclose(result, expected_rms, atol=1e-14)


# ======================================================================
# Max-Abs Computation Tests
# ======================================================================


@pytest.mark.parametrize(
    "error, expected_max",
    [
        (np.zeros(5), 0.0),
        (np.ones(4), 1.0),
        (np.array([3.0, -4.0]), 4.0),
        (np.array([-7.0, 2.0, 5.0]), 7.0),
        (np.array([[1.0, -9.0], [3.0, 4.0]]), 9.0),
    ],
    ids=["zeros", "ones", "neg-dominates", "first-element", "2x2-matrix"],
)
def test_max_abs_static(error: np.ndarray, expected_max: float) -> None:
    """Verify CalculationErrorMetric.max_abs returns correct values.

    The max-abs metric takes the absolute value of every element in the
    (possibly multidimensional) error array and returns the maximum.
    We test with zero arrays, uniform arrays, vectors with negative
    dominant elements, and 2D matrices to ensure the abs-then-max
    logic handles signs and shapes correctly.

    :param error: Input error array (vector or matrix), provided by parametrize.
    :type error: np.ndarray
    :param expected_max: Expected max-abs value for the given error, provided by parametrize.
    :type expected_max: float
    """
    # result is the scalar max-abs value computed by the static method
    result = CalculationErrorMetric.max_abs(error)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, expected_max, atol=1e-14)


@pytest.mark.parametrize(
    "error, expected_max",
    [
        (np.zeros(5), 0.0),
        (np.array([3.0, -4.0]), 4.0),
        (np.array([-7.0, 2.0, 5.0]), 7.0),
    ],
    ids=["zeros", "neg-dominates", "first-element"],
)
def test_compute_dispatches_to_max_abs(
    error: np.ndarray, expected_max: float
) -> None:
    """compute() with 'max_abs' metric should dispatch to max_abs().

    When the metric attribute is 'max_abs', compute() must delegate to
    the static max_abs() method.  The result should match the static
    call exactly, verifying that the dispatch branch for 'max_abs'
    is wired correctly.

    :param error: Input error array, provided by parametrize.
    :type error: np.ndarray
    :param expected_max: Expected max-abs value, provided by parametrize.
    :type expected_max: float
    """
    metric = CalculationErrorMetric(metric="max_abs")
    # result should equal the static max_abs() output
    result = metric.compute(error)
    np.testing.assert_allclose(result, expected_max, atol=1e-14)
