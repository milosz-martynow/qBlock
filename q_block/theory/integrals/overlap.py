"""Overlap matrix computation for Gaussian basis functions.

This module computes the overlap matrix :math:`S` between contracted
Gaussian-type orbitals (CGTOs):

.. math::

    S_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\, \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

where :math:`\\phi_\\mu` is a contracted Gaussian basis function.

For primitive Gaussians centered at :math:`\\mathbf{A}` and :math:`\\mathbf{B}`
with exponents :math:`\\alpha` and :math:`\\beta`:

.. math::

    \\langle g_a | g_b \\rangle = N_a\\, N_b\\,
        \\left(\\frac{\\pi}{\\alpha + \\beta}\\right)^{3/2}
        e^{-\\frac{\\alpha \\beta}{\\alpha + \\beta}|\\mathbf{A} - \\mathbf{B}|^2}
        \\prod_{i=x,y,z} S_i(l_{ai}, l_{bi})

where :math:`S_i` are 1D overlap integrals computed via the Obara-Saika
recurrence relations.

Classes
-------
OverlapMatrix
    Computes and stores the full overlap matrix for a molecular basis.
"""

import math
from typing import List, Tuple, Optional

import numpy as np

from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.utils import normalization_constant, get_cartesian_components


# ======================================================================
# Main overlap matrix class
# ======================================================================


class OverlapMatrix:
    """Computes and stores the overlap matrix S for a molecular basis.

    The overlap matrix elements are:

    .. math::

        S_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\, \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

    where :math:`\\phi_\\mu` are contracted Gaussian basis functions.

    The matrix is symmetric: :math:`S_{\\mu\\nu} = S_{\\nu\\mu}`.

    :param cgtos: List of contracted Gaussian-type orbitals.
    :type cgtos: List[ContractedGaussianTypeOrbital]

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
        The basis functions used to construct the matrix.
    n_basis : int
        Total number of basis functions (counting all angular components).
    matrix : np.ndarray
        The computed overlap matrix of shape ``(n_basis, n_basis)``.

    Example
    -------
    >>> from q_block import Molecule
    >>> from q_block.io.input_data import InputData
    >>> from q_block.io.basis_set import Pople
    >>> from q_block.theory.integrals import OverlapMatrix
    >>>
    >>> basis = Pople(filepath="data/basis_set/sto_gaussian_format/STO-3G.gbs")
    >>> inp = InputData()
    >>> inp.from_script([["H", 0.0, 0.0, 0.0, basis], ["H", 0.0, 0.0, 1.4, basis]])
    >>> mol = Molecule(input_data=inp)
    >>> mol.to_bohr()
    >>> mol.make_contracted_gaussian_type_orbital()
    >>>
    >>> S = OverlapMatrix(mol.contracted_gaussian_type_orbitals)
    >>> print(S.matrix.shape)  # (2, 2) for two s-type basis functions
    >>> print(S.matrix[0, 0])  # Should be close to 1.0 (normalized)
    """

    def __init__(self, cgtos: List[ContractedGaussianTypeOrbital]) -> None:
        if not cgtos:
            raise ValueError("Cannot build overlap matrix with empty basis set.")

        self.cgtos: List[ContractedGaussianTypeOrbital] = cgtos

        # Count total basis functions (each shell contributes 2l+1 functions)
        self.n_basis: int = sum(cgto.n_functions for cgto in cgtos)

        # Build the basis function index mapping
        self._build_basis_index_map()

        # Compute the overlap matrix
        self.matrix: np.ndarray = self._compute_overlap_matrix()

    # ==================================================================
    # Static methods for integral computation
    # ==================================================================

    @staticmethod
    def _gaussian_product_center(
        alpha: float,
        A: Tuple[float, float, float],
        beta: float,
        B: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Compute the center of the Gaussian product.

        When two Gaussians centered at :math:`\\mathbf{A}` and :math:`\\mathbf{B}`
        with exponents :math:`\\alpha` and :math:`\\beta` are multiplied, the
        result is a Gaussian centered at:

        .. math::

            \\mathbf{P} = \\frac{\\alpha \\mathbf{A} + \\beta \\mathbf{B}}{\\alpha + \\beta}

        :param alpha: Exponent of the first Gaussian.
        :type alpha: float
        :param A: Center of the first Gaussian (x, y, z).
        :type A: Tuple[float, float, float]
        :param beta: Exponent of the second Gaussian.
        :type beta: float
        :param B: Center of the second Gaussian (x, y, z).
        :type B: Tuple[float, float, float]

        :returns: Center of the product Gaussian (Px, Py, Pz).
        :rtype: Tuple[float, float, float]
        """
        gamma = alpha + beta
        Px = (alpha * A[0] + beta * B[0]) / gamma
        Py = (alpha * A[1] + beta * B[1]) / gamma
        Pz = (alpha * A[2] + beta * B[2]) / gamma
        return (Px, Py, Pz)

    @staticmethod
    def _overlap_1d(
        l1: int,
        l2: int,
        PA: float,
        PB: float,
        gamma: float,
    ) -> float:
        """Compute 1D overlap integral using Obara-Saika recursion.

        Evaluates the 1D overlap integral:

        .. math::

            S_{l_1, l_2} = \\int_{-\\infty}^{+\\infty}
                (x - A)^{l_1} (x - B)^{l_2} e^{-\\gamma (x - P)^2} dx

        using the Obara-Saika recurrence relations:

        .. math::

            S_{i+1,j} = PA \\cdot S_{i,j} + \\frac{1}{2\\gamma}(i \\cdot S_{i-1,j} + j \\cdot S_{i,j-1})

            S_{i,j+1} = PB \\cdot S_{i,j} + \\frac{1}{2\\gamma}(i \\cdot S_{i-1,j} + j \\cdot S_{i,j-1})

        :param l1: Angular momentum on center A.
        :type l1: int
        :param l2: Angular momentum on center B.
        :type l2: int
        :param PA: Distance P - A along this axis.
        :type PA: float
        :param PB: Distance P - B along this axis.
        :type PB: float
        :param gamma: Sum of exponents (alpha + beta).
        :type gamma: float

        :returns: Value of the 1D overlap integral.
        :rtype: float
        """
        # Build a 2D array to store S[i][j] for i = 0..l1, j = 0..l2
        S = [[0.0] * (l2 + 1) for _ in range(l1 + 1)]

        # Base case: S[0][0] = sqrt(pi/gamma)
        S[0][0] = math.sqrt(math.pi / gamma)

        # Build up S[i][0] using the first recurrence
        for i in range(l1):
            S[i + 1][0] = PA * S[i][0]
            if i > 0:
                S[i + 1][0] += i * S[i - 1][0] / (2.0 * gamma)

        # Build up S[i][j] for j > 0 using the second recurrence
        for j in range(l2):
            for i in range(l1 + 1):
                S[i][j + 1] = PB * S[i][j]
                if i > 0:
                    S[i][j + 1] += i * S[i - 1][j] / (2.0 * gamma)
                if j > 0:
                    S[i][j + 1] += j * S[i][j - 1] / (2.0 * gamma)

        return S[l1][l2]

    @staticmethod
    def primitive_overlap(
        alpha: float,
        A: Tuple[float, float, float],
        lx1: int,
        ly1: int,
        lz1: int,
        beta: float,
        B: Tuple[float, float, float],
        lx2: int,
        ly2: int,
        lz2: int,
    ) -> float:
        """Compute overlap integral between two primitive Gaussians.

        Evaluates:

        .. math::

            \\langle g_a | g_b \\rangle = \\int g_a(\\mathbf{r})\\, g_b(\\mathbf{r})\\, d\\mathbf{r}

        where :math:`g_a` and :math:`g_b` are normalized Cartesian Gaussian
        primitives.

        :param alpha: Exponent of the first primitive.
        :type alpha: float
        :param A: Center of the first primitive (x, y, z) in Bohr.
        :type A: Tuple[float, float, float]
        :param lx1: Angular momentum in x for the first primitive.
        :type lx1: int
        :param ly1: Angular momentum in y for the first primitive.
        :type ly1: int
        :param lz1: Angular momentum in z for the first primitive.
        :type lz1: int
        :param beta: Exponent of the second primitive.
        :type beta: float
        :param B: Center of the second primitive (x, y, z) in Bohr.
        :type B: Tuple[float, float, float]
        :param lx2: Angular momentum in x for the second primitive.
        :type lx2: int
        :param ly2: Angular momentum in y for the second primitive.
        :type ly2: int
        :param lz2: Angular momentum in z for the second primitive.
        :type lz2: int

        :returns: Overlap integral value.
        :rtype: float
        """
        gamma = alpha + beta

        # Gaussian product center
        P = OverlapMatrix._gaussian_product_center(alpha, A, beta, B)

        # Distances from product center to original centers
        PA = (P[0] - A[0], P[1] - A[1], P[2] - A[2])
        PB = (P[0] - B[0], P[1] - B[1], P[2] - B[2])

        # Squared distance between centers
        AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2

        # Pre-exponential factor
        pre_factor = math.exp(-alpha * beta * AB_sq / gamma)

        # 1D overlap integrals
        Sx = OverlapMatrix._overlap_1d(lx1, lx2, PA[0], PB[0], gamma)
        Sy = OverlapMatrix._overlap_1d(ly1, ly2, PA[1], PB[1], gamma)
        Sz = OverlapMatrix._overlap_1d(lz1, lz2, PA[2], PB[2], gamma)

        # Normalization constants
        N1 = normalization_constant(alpha, lx1, ly1, lz1)
        N2 = normalization_constant(beta, lx2, ly2, lz2)

        return N1 * N2 * pre_factor * Sx * Sy * Sz

    @staticmethod
    def contracted_overlap(
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
    ) -> float:
        """Compute overlap integral between two contracted Gaussians.

        Evaluates:

        .. math::

            S_{\\mu\\nu} = \\sum_{p}^{K_\\mu} \\sum_{q}^{K_\\nu}
                d_p d_q \\langle g_p | g_q \\rangle

        where :math:`d_p` are contraction coefficients.

        :param cgto1: First contracted Gaussian shell.
        :type cgto1: ContractedGaussianTypeOrbital
        :param lx1: Angular momentum in x for the first function.
        :type lx1: int
        :param ly1: Angular momentum in y for the first function.
        :type ly1: int
        :param lz1: Angular momentum in z for the first function.
        :type lz1: int
        :param cgto2: Second contracted Gaussian shell.
        :type cgto2: ContractedGaussianTypeOrbital
        :param lx2: Angular momentum in x for the second function.
        :type lx2: int
        :param ly2: Angular momentum in y for the second function.
        :type ly2: int
        :param lz2: Angular momentum in z for the second function.
        :type lz2: int

        :returns: Overlap integral value.
        :rtype: float
        """
        A = (cgto1.center.x, cgto1.center.y, cgto1.center.z)
        B = (cgto2.center.x, cgto2.center.y, cgto2.center.z)

        overlap = 0.0

        for p in range(cgto1.n_primitives):
            alpha = cgto1.exponents[p]
            d_p = cgto1.contractions[p]

            for q in range(cgto2.n_primitives):
                beta = cgto2.exponents[q]
                d_q = cgto2.contractions[q]

                prim_ovlp = OverlapMatrix.primitive_overlap(
                    alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2
                )
                overlap += d_p * d_q * prim_ovlp

        return overlap

    # ==================================================================
    # Instance methods
    # ==================================================================

    def _build_basis_index_map(self) -> None:
        """Build mapping from basis function index to (shell, angular component).

        Creates ``_basis_map``: a list where each entry is a tuple
        ``(shell_index, lx, ly, lz)`` identifying which shell and
        Cartesian component corresponds to each basis function index.
        """
        self._basis_map: List[Tuple[int, int, int, int]] = []

        for shell_idx, cgto in enumerate(self.cgtos):
            components = get_cartesian_components(cgto.l)
            for lx, ly, lz in components:
                self._basis_map.append((shell_idx, lx, ly, lz))

    def _compute_overlap_matrix(self) -> np.ndarray:
        """Compute the full overlap matrix.

        Uses symmetry: only computes upper triangle and mirrors to lower.

        :returns: Overlap matrix of shape ``(n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        S = np.zeros((self.n_basis, self.n_basis))

        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]

            for nu in range(mu, self.n_basis):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]

                S_mn = self.contracted_overlap(
                    cgto1, lx1, ly1, lz1, cgto2, lx2, ly2, lz2
                )

                S[mu, nu] = S_mn
                S[nu, mu] = S_mn  # Symmetry

        return S

    def __repr__(self) -> str:
        return f"OverlapMatrix(n_basis={self.n_basis})"

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Access matrix element by index.

        :param key: Tuple of (row, column) indices.
        :type key: Tuple[int, int]

        :returns: Matrix element S[row, col].
        :rtype: float
        """
        return self.matrix[key]
