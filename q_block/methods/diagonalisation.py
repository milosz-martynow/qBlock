r"""Diagonalisation routines for the generalised eigenvalue problem.

This module provides functions for solving the Roothaan–Hall eigenvalue
problem that appears at the core of every SCF iteration:

.. math::

    \mathbf{F}\,\mathbf{C} = \mathbf{S}\,\mathbf{C}\,\boldsymbol{\varepsilon}

Instead of solving the generalised problem directly, the Fock matrix is
first transformed to an orthogonal basis using the symmetric
orthogonalisation matrix :math:`\mathbf{X} = \mathbf{S}^{-1/2}`:

.. math::

    \mathbf{F}' = \mathbf{X}^T \mathbf{F}\, \mathbf{X}

The standard eigenvalue problem :math:`\mathbf{F}'\mathbf{C}'
= \mathbf{C}'\boldsymbol{\varepsilon}` is then solved, and the MO
coefficients are back-transformed to the AO basis:

.. math::

    \mathbf{C} = \mathbf{X}\,\mathbf{C}'

Functions
---------
diagonalise_fock
    Diagonalise one or more Fock matrices in the orthogonal basis.
diagonalise_single
    Diagonalise a single Fock matrix.
"""

from typing import Tuple, Union

import numpy as np
from scipy.linalg import eigh


def diagonalise_single(
    fock: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Diagonalise a single Fock matrix in the orthogonal basis.

    Transforms the Fock matrix to the orthogonal basis, solves the
    standard eigenvalue problem, and back-transforms the MO coefficients.

    .. math::

        \mathbf{F}' = \mathbf{X}^T \mathbf{F}\, \mathbf{X}

        \mathbf{F}' \mathbf{C}' = \mathbf{C}' \boldsymbol{\varepsilon}

        \mathbf{C} = \mathbf{X}\, \mathbf{C}'

    :param fock: Fock matrix in the AO basis, shape
        ``(n_basis, n_basis)``.
    :type fock: np.ndarray
    :param X: Orthogonalisation matrix :math:`\mathbf{S}^{-1/2}`, shape
        ``(n_basis, n_basis)``.
    :type X: np.ndarray
    :returns: ``(C, epsilon)`` — MO coefficient matrix and orbital
        energies, both in the AO basis.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    F_prime = X.T @ fock @ X
    epsilon, C_prime = eigh(F_prime)
    C = X @ C_prime
    return C, epsilon


def diagonalise_fock(
    fock: Union[np.ndarray, Tuple[np.ndarray, ...]],
    X: np.ndarray,
) -> Tuple[
    Union[np.ndarray, Tuple[np.ndarray, ...]],
    Union[np.ndarray, Tuple[np.ndarray, ...]],
]:
    r"""Diagonalise one or more Fock matrices in the orthogonal basis.

    For restricted methods *fock* is a single matrix; for unrestricted
    methods it is a tuple (e.g.
    :math:`(\mathbf{F}^\alpha, \mathbf{F}^\beta)`).
    Each matrix is diagonalised independently via
    :func:`diagonalise_single`.

    :param fock: Fock matrix or tuple of Fock matrices.
    :type fock: Union[np.ndarray, Tuple[np.ndarray, ...]]
    :param X: Orthogonalisation matrix :math:`\mathbf{S}^{-1/2}`.
    :type X: np.ndarray
    :returns: ``(C, epsilon)`` — MO coefficients and orbital energies,
        each a single array or a tuple matching *fock*.
    :rtype: Tuple
    """
    if isinstance(fock, tuple):
        results = [diagonalise_single(f, X) for f in fock]
        C_tuple = tuple(r[0] for r in results)
        eps_tuple = tuple(r[1] for r in results)
        return C_tuple, eps_tuple
    return diagonalise_single(fock, X)
