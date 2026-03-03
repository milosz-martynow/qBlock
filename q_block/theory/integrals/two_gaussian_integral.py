"""Abstract base class for two-center Gaussian integral matrices.

This module provides the framework for computing integral matrices
between contracted Gaussian-type orbitals (CGTOs). All one-electron integrals
over Gaussians share a common mathematical structure:

.. math::

    I_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\, K(\\mathbf{r})\\,
        \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

where :math:`K(\\mathbf{r})` is the kernel operator:

- **Overlap**: :math:`K = 1`
- **Kinetic energy**: :math:`K = -\\frac{1}{2}\\nabla^2`
- **Nuclear attraction**: :math:`K = -\\sum_C \\frac{Z_C}{|\\mathbf{r} - \\mathbf{R}_C|}`

Subclasses implement the specific kernel by overriding the
:meth:`_compute_element` method.

Classes
-------
TwoGaussianIntegral
    Abstract base class for all two-center Gaussian integral matrices.
"""

import itertools
import math
from abc import abstractmethod
from typing import Callable, List, Tuple

import numpy as np

from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.integrals.integral import Integral
from q_block.theory.utils import get_cartesian_components


class TwoGaussianIntegral(Integral):
    """Abstract base class for two-center Gaussian integral matrices.

    Inherits from :class:`Integral` and provides infrastructure specific
    to Gaussian-type orbital integrals:

    - Basis function index mapping for contracted Gaussians
    - Gaussian product center computation (Gaussian product theorem)
    - 1D overlap integrals via Obara-Saika recursion
    - Symmetric matrix computation with upper-triangle optimization

    Subclasses must implement :meth:`_compute_element` to define the
    specific integral kernel.

    :param cgtos: List of contracted Gaussian-type orbitals.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param kwargs: Additional parameters for specific integral types.

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
        The basis functions used to construct the matrix.
    n_basis : int
        Total number of basis functions (counting all angular components).
    matrix : np.ndarray
        The computed integral matrix of shape ``(n_basis, n_basis)``.

    Notes
    -----
    All integral matrices are symmetric: :math:`I_{\\mu\\nu} = I_{\\nu\\mu}`.
    The matrix computation exploits this symmetry by only computing the
    upper triangle.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        **kwargs,
    ) -> None:
        if not cgtos:
            raise ValueError(
                f"Cannot build {self.__class__.__name__} matrix with empty basis set."
            )

        self.cgtos: List[ContractedGaussianTypeOrbital] = cgtos

        # Count total basis functions (each shell contributes 2l+1 functions)
        self.n_basis: int = sum(cgto.n_functions for cgto in cgtos)

        # Build the basis function index mapping
        self._build_basis_index_map()

        # Store additional parameters (e.g., nuclei for NuclearAttraction)
        self._init_params(**kwargs)

        # Compute the integral matrix
        self.matrix: np.ndarray = self._compute_matrix()

    def _init_params(self, **kwargs) -> None:
        """Initialize additional parameters for specific integral types.

        Subclasses can override this to store extra parameters like
        nuclear positions for NuclearAttraction. Default implementation
        does nothing.

        :param kwargs: Additional keyword arguments.
        """
        pass

    # ==================================================================
    # Static utility methods for Gaussian integrals
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
    def _contract_primitives(
        cgtos: List[ContractedGaussianTypeOrbital],
        angular_momenta: List[Tuple[int, int, int]],
        primitive_kernel: Callable[
            [
                List[float],
                List[Tuple[float, float, float]],
                List[Tuple[int, int, int]],
            ],
            float,
        ],
    ) -> float:
        """Compute contracted integral over n Gaussian centers.

        This generic method handles any number of contracted Gaussians by:

        1. Extracting centers, exponents, and coefficients from each CGTO
        2. Iterating over all primitive combinations (Cartesian product)
        3. Accumulating weighted primitive integrals

        .. math::

            I = \\sum_{p_1}^{K_1} \\sum_{p_2}^{K_2} \\cdots \\sum_{p_n}^{K_n}
                d_{p_1} d_{p_2} \\cdots d_{p_n} \\cdot
                \\text{kernel}(\\alpha_{p_1}, \\alpha_{p_2}, \\ldots)

        :param cgtos: List of n contracted Gaussian-type orbitals.
        :type cgtos: List[ContractedGaussianTypeOrbital]
        :param angular_momenta: List of (lx, ly, lz) tuples for each CGTO.
        :type angular_momenta: List[Tuple[int, int, int]]
        :param primitive_kernel: Function that computes primitive integral.
            Signature: kernel(exponents, centers, angular_momenta) -> float
        :type primitive_kernel: Callable

        :returns: Contracted integral value.
        :rtype: float
        """
        # Extract centers from each CGTO
        centers = [
            (cgto.center.x, cgto.center.y, cgto.center.z) for cgto in cgtos
        ]

        # Build iteration ranges for each primitive
        primitive_ranges = [range(cgto.n_primitives) for cgto in cgtos]

        integral = 0.0

        # Iterate over all primitive combinations (Cartesian product)
        for indices in itertools.product(*primitive_ranges):
            # Collect exponents and coefficients for this combination
            exponents = [
                cgtos[i].exponents[idx] for i, idx in enumerate(indices)
            ]
            coefficients = [
                cgtos[i].contractions[idx] for i, idx in enumerate(indices)
            ]

            # Compute product of contraction coefficients
            coeff_product = 1.0
            for c in coefficients:
                coeff_product *= c

            # Compute primitive integral using the kernel
            prim_integral = primitive_kernel(exponents, centers, angular_momenta)

            integral += coeff_product * prim_integral

        return integral

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
        # Handle negative angular momentum (needed for kinetic energy recursion)
        if l1 < 0 or l2 < 0:
            return 0.0

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

    # ==================================================================
    # Basis index mapping
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

    # ==================================================================
    # Abstract method for specific integral computation
    # ==================================================================

    @abstractmethod
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
        """Compute a single matrix element between two basis functions.

        Subclasses must implement this method to define the specific
        integral kernel. This method computes:

        .. math::

            I_{\\mu\\nu} = \\int \\phi_\\mu(\\mathbf{r})\\, K(\\mathbf{r})\\,
                \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

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

        :returns: Integral matrix element value.
        :rtype: float
        """
        pass

    # ==================================================================
    # Matrix computation
    # ==================================================================

    def _compute_matrix(self) -> np.ndarray:
        """Compute the full integral matrix.

        Uses symmetry: only computes upper triangle and mirrors to lower.
        Calls :meth:`_compute_element` for each unique pair.

        :returns: Integral matrix of shape ``(n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        matrix = np.zeros((self.n_basis, self.n_basis))

        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]

            for nu in range(mu, self.n_basis):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]

                element = self._compute_element(
                    cgto1, lx1, ly1, lz1, cgto2, lx2, ly2, lz2
                )

                matrix[mu, nu] = element
                matrix[nu, mu] = element  # Symmetry

        return matrix
