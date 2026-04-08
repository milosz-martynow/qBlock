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
Overlap
    Computes and stores the full overlap matrix for a molecular basis.
"""

import math
from typing import List, Tuple

from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.integrals.two_gaussian_integral import TwoGaussianIntegral
from q_block.theory.utils import normalization_constant


class Overlap(TwoGaussianIntegral):
    """Computes and stores the overlap matrix S for a molecular basis.

    Inherits from :class:`TwoGaussianIntegral` to reuse common infrastructure
    for basis indexing, matrix computation, and Gaussian utilities.

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
    """

    # ==================================================================
    # Static methods for overlap computation
    # ==================================================================

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
        P = TwoGaussianIntegral._gaussian_product_center(alpha, A, beta, B)

        # Distances from product center to original centers
        PA = (P[0] - A[0], P[1] - A[1], P[2] - A[2])
        PB = (P[0] - B[0], P[1] - B[1], P[2] - B[2])

        # Squared distance between centers
        AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2

        # Pre-exponential factor
        pre_factor = math.exp(-alpha * beta * AB_sq / gamma)

        # 1D overlap integrals
        Sx = TwoGaussianIntegral._overlap_1d(lx1, lx2, PA[0], PB[0], gamma)
        Sy = TwoGaussianIntegral._overlap_1d(ly1, ly2, PA[1], PB[1], gamma)
        Sz = TwoGaussianIntegral._overlap_1d(lz1, lz2, PA[2], PB[2], gamma)

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

                prim_ovlp = Overlap.primitive_overlap(
                    alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2
                )
                overlap += d_p * d_q * prim_ovlp

        return overlap

    # ==================================================================
    # Abstract method implementation
    # ==================================================================

    def _compute_element(
        self,
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
    ) -> float:
        """Compute overlap matrix element.

        Implements the abstract method from :class:`Integral`.

        :returns: Overlap integral value S_μν.
        :rtype: float
        """
        return self.contracted_overlap(
            cgto1, lx1, ly1, lz1, cgto2, lx2, ly2, lz2
        )
