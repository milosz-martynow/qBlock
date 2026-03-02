"""Kinetic energy matrix computation for Gaussian basis functions.

This module computes the kinetic energy matrix :math:`T` between contracted
Gaussian-type orbitals (CGTOs):

.. math::

    T_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\,
        \\left(-\\frac{1}{2}\\nabla^2\\right)\\,
        \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

where :math:`\\phi_\\mu` is a contracted Gaussian basis function.

Using integration by parts, this becomes:

.. math::

    T_{\\mu\\nu} = \\frac{1}{2} \\int
        \\nabla\\phi_\\mu(\\mathbf{r}) \\cdot \\nabla\\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

For primitive Gaussians, the kinetic energy integral is computed as a sum
of overlap-like integrals with modified angular momentum indices.

Classes
-------
KineticEnergy
    Computes and stores the full kinetic energy matrix for a molecular basis.
"""

import math
from typing import List, Tuple

import numpy as np

from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.integrals.overlap import Overlap
from q_block.theory.utils import normalization_constant


# ======================================================================
# Main kinetic energy matrix class
# ======================================================================


class KineticEnergy(Overlap):
    """Computes and stores the kinetic energy matrix T for a molecular basis.

    Inherits from :class:`Overlap` to reuse Gaussian product center
    computation and 1D overlap integral methods.

    The kinetic energy matrix elements are:

    .. math::

        T_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\,
            \\left(-\\frac{1}{2}\\nabla^2\\right)\\,
            \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

    where :math:`\\phi_\\mu` are contracted Gaussian basis functions.

    The matrix is symmetric: :math:`T_{\\mu\\nu} = T_{\\nu\\mu}`.

    :param cgtos: List of contracted Gaussian-type orbitals.
    :type cgtos: List[ContractedGaussianTypeOrbital]

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
        The basis functions used to construct the matrix.
    n_basis : int
        Total number of basis functions (counting all angular components).
    matrix : np.ndarray
        The computed kinetic energy matrix of shape ``(n_basis, n_basis)``.
    """

    def __init__(self, cgtos: List[ContractedGaussianTypeOrbital]) -> None:
        if not cgtos:
            raise ValueError("Cannot build kinetic energy matrix with empty basis set.")

        self.cgtos: List[ContractedGaussianTypeOrbital] = cgtos

        # Count total basis functions (each shell contributes 2l+1 functions)
        self.n_basis: int = sum(cgto.n_functions for cgto in cgtos)

        # Build the basis function index mapping (inherited from Overlap)
        self._build_basis_index_map()

        # Compute the kinetic energy matrix
        self.matrix: np.ndarray = self._compute_kinetic_matrix()

    # ==================================================================
    # Static methods for kinetic energy computation
    # ==================================================================

    @staticmethod
    def _kinetic_1d(
        l1: int,
        l2: int,
        PA: float,
        PB: float,
        gamma: float,
        alpha: float,
        beta: float,
    ) -> float:
        """Compute 1D kinetic energy integral.

        The 1D kinetic energy integral is computed from the derivative formula:

        .. math::

            T_{l_1, l_2} = l_1 l_2 S_{l_1-1, l_2-1}
                         - 2\\alpha l_2 S_{l_1+1, l_2-1}
                         - 2\\beta l_1 S_{l_1-1, l_2+1}
                         + 4\\alpha\\beta S_{l_1+1, l_2+1}

        where :math:`S_{i,j}` is the 1D overlap integral.

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
        :param alpha: Exponent of the first Gaussian.
        :type alpha: float
        :param beta: Exponent of the second Gaussian.
        :type beta: float

        :returns: Value of the 1D kinetic energy integral.
        :rtype: float
        """
        # Term 1: l1 * l2 * S(l1-1, l2-1)
        term1 = l1 * l2 * Overlap._overlap_1d(l1 - 1, l2 - 1, PA, PB, gamma)

        # Term 2: -2 * alpha * l2 * S(l1+1, l2-1)
        term2 = -2.0 * alpha * l2 * Overlap._overlap_1d(l1 + 1, l2 - 1, PA, PB, gamma)

        # Term 3: -2 * beta * l1 * S(l1-1, l2+1)
        term3 = -2.0 * beta * l1 * Overlap._overlap_1d(l1 - 1, l2 + 1, PA, PB, gamma)

        # Term 4: 4 * alpha * beta * S(l1+1, l2+1)
        term4 = 4.0 * alpha * beta * Overlap._overlap_1d(l1 + 1, l2 + 1, PA, PB, gamma)

        return term1 + term2 + term3 + term4

    @staticmethod
    def primitive_kinetic(
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
        """Compute kinetic energy integral between two primitive Gaussians.

        Evaluates:

        .. math::

            \\langle g_a | -\\frac{1}{2}\\nabla^2 | g_b \\rangle

        where :math:`g_a` and :math:`g_b` are normalized Cartesian Gaussian
        primitives. The integral is computed as:

        .. math::

            T = \\frac{1}{2} N_a N_b e^{-\\mu R_{AB}^2}
                (T_x S_y S_z + S_x T_y S_z + S_x S_y T_z)

        where :math:`\\mu = \\frac{\\alpha\\beta}{\\alpha+\\beta}`.

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

        :returns: Kinetic energy integral value.
        :rtype: float
        """
        gamma = alpha + beta

        # Gaussian product center (inherited from Overlap)
        P = Overlap._gaussian_product_center(alpha, A, beta, B)

        # Distances from product center to original centers
        PA = (P[0] - A[0], P[1] - A[1], P[2] - A[2])
        PB = (P[0] - B[0], P[1] - B[1], P[2] - B[2])

        # Squared distance between centers
        AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2

        # Pre-exponential factor
        pre_factor = math.exp(-alpha * beta * AB_sq / gamma)

        # 1D overlap integrals (inherited from Overlap)
        Sx = Overlap._overlap_1d(lx1, lx2, PA[0], PB[0], gamma)
        Sy = Overlap._overlap_1d(ly1, ly2, PA[1], PB[1], gamma)
        Sz = Overlap._overlap_1d(lz1, lz2, PA[2], PB[2], gamma)

        # 1D kinetic energy integrals
        Tx = KineticEnergy._kinetic_1d(lx1, lx2, PA[0], PB[0], gamma, alpha, beta)
        Ty = KineticEnergy._kinetic_1d(ly1, ly2, PA[1], PB[1], gamma, alpha, beta)
        Tz = KineticEnergy._kinetic_1d(lz1, lz2, PA[2], PB[2], gamma, alpha, beta)

        # Normalization constants
        N1 = normalization_constant(alpha, lx1, ly1, lz1)
        N2 = normalization_constant(beta, lx2, ly2, lz2)

        # Total kinetic energy: T = (1/2) * (Tx*Sy*Sz + Sx*Ty*Sz + Sx*Sy*Tz)
        kinetic = 0.5 * (Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz)

        return N1 * N2 * pre_factor * kinetic

    @staticmethod
    def contracted_kinetic(
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
    ) -> float:
        """Compute kinetic energy integral between two contracted Gaussians.

        Evaluates:

        .. math::

            T_{\\mu\\nu} = \\sum_{p}^{K_\\mu} \\sum_{q}^{K_\\nu}
                d_p d_q \\langle g_p | -\\frac{1}{2}\\nabla^2 | g_q \\rangle

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

        :returns: Kinetic energy integral value.
        :rtype: float
        """
        A = (cgto1.center.x, cgto1.center.y, cgto1.center.z)
        B = (cgto2.center.x, cgto2.center.y, cgto2.center.z)

        kinetic = 0.0

        for p in range(cgto1.n_primitives):
            alpha = cgto1.exponents[p]
            d_p = cgto1.contractions[p]

            for q in range(cgto2.n_primitives):
                beta = cgto2.exponents[q]
                d_q = cgto2.contractions[q]

                prim_kin = KineticEnergy.primitive_kinetic(
                    alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2
                )
                kinetic += d_p * d_q * prim_kin

        return kinetic

    # ==================================================================
    # Instance methods
    # ==================================================================

    def _compute_kinetic_matrix(self) -> np.ndarray:
        """Compute the full kinetic energy matrix.

        Uses symmetry: only computes upper triangle and mirrors to lower.

        :returns: Kinetic energy matrix of shape ``(n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        T = np.zeros((self.n_basis, self.n_basis))

        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]

            for nu in range(mu, self.n_basis):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]

                T_mn = self.contracted_kinetic(
                    cgto1, lx1, ly1, lz1, cgto2, lx2, ly2, lz2
                )

                T[mu, nu] = T_mn
                T[nu, mu] = T_mn  # Symmetry

        return T

    def __repr__(self) -> str:
        return f"KineticEnergy(n_basis={self.n_basis})"

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Access matrix element by index.

        :param key: Tuple of (row, column) indices.
        :type key: Tuple[int, int]

        :returns: Matrix element T[row, col].
        :rtype: float
        """
        return self.matrix[key]
