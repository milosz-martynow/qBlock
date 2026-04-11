r"""SCF convergence error measurement.

This module provides a configurable error metric used to assess
convergence during the SCF loop.

Classes
-------
CalculationErrorMetric
    Computes a scalar convergence metric from an error vector.
"""

import numpy as np


class CalculationErrorMetric:
    """Configurable scalar error metric for SCF convergence.

    Instantiated once before the SCF loop begins.  During each
    iteration the :meth:`compute` method is called with the current
    error vector to obtain a scalar that is compared against the
    convergence threshold.

    :param metric: Name of the reduction method to use.
        Supported values: ``"rms"`` (root-mean-square, default),
        ``"max_abs"`` (maximum absolute value).
    :type metric: str

    Attributes
    ----------
    metric : str
        Name of the active reduction method.
    """

    SUPPORTED_METRICS = ("rms", "max_abs")

    def __init__(self, metric: str = "rms") -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}; "
                f"supported: {self.SUPPORTED_METRICS}."
            )
        self.metric: str = metric

    def compute(self, error: np.ndarray) -> float:
        """Reduce an error vector to a scalar convergence measure.

        Dispatches to the method selected at construction time.

        :param error: Error matrix or vector (e.g. the DIIS
            commutator).
        :type error: np.ndarray
        :returns: Scalar error measure.
        :rtype: float
        """
        if self.metric == "rms":
            return self.rms(error)
        return self.max_abs(error)

    @staticmethod
    def rms(error: np.ndarray) -> float:
        r"""Root-mean-square of the error.

        .. math::

            \text{RMS} = \sqrt{\frac{1}{N} \sum_{i}^{N} e_i^2}

        :param error: Error array.
        :type error: np.ndarray
        :returns: RMS value.
        :rtype: float
        """
        flat = error.ravel()
        return float(np.sqrt(np.dot(flat, flat) / flat.size))

    @staticmethod
    def max_abs(error: np.ndarray) -> float:
        """Maximum absolute value of the error.

        :param error: Error array.
        :type error: np.ndarray
        :returns: Max-abs value.
        :rtype: float
        """
        return float(np.max(np.abs(error)))

    def __repr__(self) -> str:
        return f"CalculationErrorMetric(metric={self.metric!r})"
