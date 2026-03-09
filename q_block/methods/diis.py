r"""DIIS convergence accelerator for SCF procedures.

This module implements Pulay's Direct Inversion of the Iterative Subspace
(DIIS) algorithm, a convergence accelerator widely used in self-consistent
field procedures.

At each SCF iteration the current Fock matrix and an error vector are
stored.  When enough vectors have been accumulated, DIIS constructs
an improved Fock matrix as an optimal linear combination of the stored
matrices by minimising the norm of the interpolated error:

.. math::

    \mathbf{F}_{DIIS} = \sum_{i=1}^{m} c_i \, \mathbf{F}_i

subject to the constraint :math:`\sum_i c_i = 1`.  The coefficients
:math:`c_i` are obtained by solving the Lagrangian linear system:

.. math::

    \begin{pmatrix}
        B_{11}   & \cdots & B_{1m}   & -1 \\
        \vdots   & \ddots & \vdots   & \vdots \\
        B_{m1}   & \cdots & B_{mm}   & -1 \\
        -1       & \cdots & -1       &  0
    \end{pmatrix}
    \begin{pmatrix} c_1 \\ \vdots \\ c_m \\ \lambda \end{pmatrix}
    =
    \begin{pmatrix} 0 \\ \vdots \\ 0 \\ -1 \end{pmatrix}

where :math:`B_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j`.

The error vector is typically the commutator of the Fock and density
matrices in the AO basis:

.. math::

    \mathbf{e} = \mathbf{F}\mathbf{P}\mathbf{S}
                - \mathbf{S}\mathbf{P}\mathbf{F}

Classes
-------
DIIS
    Stores Fock/error history and performs DIIS extrapolation.
"""

from typing import List, Tuple, Union

import numpy as np


class DIIS:
    """Pulay DIIS convergence accelerator.

    Accumulates Fock matrices and their associated error vectors over
    successive SCF iterations.  Once at least two pairs have been stored,
    :meth:`extrapolate` returns an optimal linear combination that
    accelerates convergence.

    :param max_vectors: Maximum number of Fock/error pairs to retain.
        When this limit is reached the oldest pair is discarded.
    :type max_vectors: int

    Attributes
    ----------
    max_vectors : int
        Maximum DIIS subspace size.
    """

    def __init__(self, max_vectors: int = 6) -> None:
        if max_vectors < 2:
            raise ValueError(
                f"max_vectors must be >= 2; got {max_vectors}."
            )
        self.max_vectors: int = max_vectors
        self._fock_vectors: List[np.ndarray] = []
        self._error_vectors: List[np.ndarray] = []

    # ==================================================================
    # Public interface
    # ==================================================================

    def extrapolate(
        self,
        fock: Union[np.ndarray, Tuple[np.ndarray, ...]],
        error: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        r"""Store the current Fock/error pair and return an extrapolated
        Fock matrix.

        If fewer than two pairs have been accumulated the original
        *fock* is returned unchanged.

        :param fock: Current Fock matrix, or tuple of Fock matrices
            (e.g. :math:`(\mathbf{F}^\alpha, \mathbf{F}^\beta)` for
            unrestricted methods).
        :type fock: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param error: DIIS error vector (flattened).
        :type error: np.ndarray
        :returns: Extrapolated Fock matrix (same type as *fock*).
        :rtype: Union[np.ndarray, Tuple[np.ndarray, ...]]
        """
        is_tuple = isinstance(fock, tuple)
        fock_flat = (
            np.concatenate([f.ravel() for f in fock]) if is_tuple
            else fock.ravel()
        )

        # Store vectors; trim to max size
        self._fock_vectors.append(fock_flat.copy())
        self._error_vectors.append(error.ravel().copy())
        if len(self._fock_vectors) > self.max_vectors:
            self._fock_vectors.pop(0)
            self._error_vectors.pop(0)

        n = len(self._error_vectors)
        if n < 2:
            return fock

        # Build B matrix
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = np.dot(
                    self._error_vectors[i],
                    self._error_vectors[j],
                )
        B[:n, n] = B[n, :n] = -1.0

        # Right-hand side
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0

        # Solve for DIIS coefficients
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return fock

        # Extrapolate Fock matrix
        fock_new_flat = sum(
            coeffs[i] * self._fock_vectors[i] for i in range(n)
        )

        if is_tuple:
            single_size = fock[0].size
            n_matrices = len(fock)
            fock_new = tuple(
                fock_new_flat[k * single_size : (k + 1) * single_size].reshape(
                    fock[k].shape
                )
                for k in range(n_matrices)
            )
            return fock_new
        return fock_new_flat.reshape(fock.shape)

    @staticmethod
    def compute_error(
        F: Union[np.ndarray, Tuple[np.ndarray, ...]],
        P: Union[np.ndarray, Tuple[np.ndarray, ...]],
        S: np.ndarray,
    ) -> np.ndarray:
        r"""Compute the standard SCF commutator error.

        .. math::

            \mathbf{e} = \mathbf{F}\mathbf{P}\mathbf{S}
                        - \mathbf{S}\mathbf{P}\mathbf{F}

        For unrestricted methods (when *F* and *P* are tuples), the
        commutator is computed for each spin channel and the flattened
        results are concatenated.

        :param F: Fock matrix (or tuple for unrestricted).
        :type F: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param P: Density matrix (or tuple for unrestricted).
        :type P: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param S: Overlap matrix, shape ``(n_basis, n_basis)``.
        :type S: np.ndarray
        :returns: Commutator error (matrix or concatenated vector).
        :rtype: np.ndarray
        """
        if isinstance(F, tuple):
            errors = [f @ p @ S - S @ p @ f for f, p in zip(F, P)]
            return np.concatenate([e.ravel() for e in errors])
        return F @ P @ S - S @ P @ F

    def reset(self) -> None:
        """Clear all stored Fock/error vectors."""
        self._fock_vectors.clear()
        self._error_vectors.clear()

    def __repr__(self) -> str:
        return (
            f"DIIS(max_vectors={self.max_vectors}, "
            f"stored={len(self._fock_vectors)})"
        )
