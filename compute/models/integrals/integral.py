"""Abstract base class for integral matrices.

This module provides the most generic framework for computing integral matrices.
All integral types share a common structure:

.. math::

    I_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\, K(\\mathbf{r})\\,
        \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

where :math:`K(\\mathbf{r})` is the kernel operator.

This abstract base class defines the interface without assuming any specific
basis function type (Gaussian, Slater, plane wave, etc.).

Classes
-------
Integral
    Abstract base class for all integral matrices.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Integral(ABC):
    """Abstract base class for integral matrices.

    Provides the minimal interface for integral matrix classes:

    - ``n_basis``: Number of basis functions
    - ``matrix``: The computed integral matrix
    - ``_compute_matrix()``: Abstract method to compute the matrix

    Subclasses implement the specific basis function type and integral
    computation algorithm.

    Attributes
    ----------
    n_basis : int
        Total number of basis functions.
    matrix : np.ndarray
        The computed integral matrix of shape ``(n_basis, n_basis)``.

    Notes
    -----
    All integral matrices are assumed symmetric: :math:`I_{\\mu\\nu} = I_{\\nu\\mu}`.
    """

    n_basis: int
    matrix: np.ndarray

    # ==================================================================
    # Abstract method for matrix computation
    # ==================================================================

    @abstractmethod
    def _compute_matrix(self) -> np.ndarray:
        """Compute the full integral matrix.

        Subclasses must implement this method to define how the matrix
        is computed based on the specific basis function type.

        :returns: Integral matrix of shape ``(n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        pass

    # ==================================================================
    # Magic methods
    # ==================================================================

    def __repr__(self) -> str:
        """String representation showing class name and basis size."""
        return f"{self.__class__.__name__}(n_basis={self.n_basis})"

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Access matrix element by index.

        :param key: Tuple of (row, column) indices.
        :type key: Tuple[int, int]

        :returns: Matrix element I[row, col].
        :rtype: float
        """
        return self.matrix[key]

    def __array__(self) -> np.ndarray:
        """Return the matrix as a NumPy array.

        Enables ``np.array(integral)`` conversion.

        :returns: The integral matrix.
        :rtype: np.ndarray
        """
        return self.matrix
