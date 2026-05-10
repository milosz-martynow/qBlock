"""Two-electron repulsion integral computation for Gaussian basis functions.

This module computes the two-electron repulsion integrals (ERI) between
contracted Gaussian-type orbitals (CGTOs):

.. math::

    (\\mu\\nu|\\lambda\\sigma) = \\int\\int \\phi_\\mu(\\mathbf{r}_1)\\,
        \\phi_\\nu(\\mathbf{r}_1)\\,
        \\frac{1}{|\\mathbf{r}_1 - \\mathbf{r}_2|}\\,
        \\phi_\\lambda(\\mathbf{r}_2)\\,
        \\phi_\\sigma(\\mathbf{r}_2)\\, d\\mathbf{r}_1\\, d\\mathbf{r}_2

where :math:`\\phi_\\mu` are contracted Gaussian basis functions.

The Coulomb integral over four Gaussians is evaluated using the Boys function
:math:`F_n(x)` and the McMurchie-Davidson recurrence relations for
Hermite Coulomb integrals.

Classes
-------
TwoElectronRepulsion
    Computes and stores the full ERI tensor for a molecular basis.
"""

import math
from typing import List, Tuple

import numpy as np

from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.models.integrals.two_gaussian_integral import TwoGaussianIntegral
from q_block.compute.utilities.mathematics import (
    boys_function,
    hermite_coulomb_table,
    hermite_expansion_coefficients,
    normalization_constant,
)


class TwoElectronRepulsion(TwoGaussianIntegral):
    """Computes and stores the two-electron repulsion integrals for a molecular basis.

    Inherits from :class:`TwoGaussianIntegral` to reuse common infrastructure
    for basis indexing and Gaussian utilities.

    The two-electron repulsion integral (ERI) tensor elements are:

    .. math::

        (\\mu\\nu|\\lambda\\sigma) = \\int\\int \\phi_\\mu(\\mathbf{r}_1)\\,
            \\phi_\\nu(\\mathbf{r}_1)\\,
            \\frac{1}{|\\mathbf{r}_1 - \\mathbf{r}_2|}\\,
            \\phi_\\lambda(\\mathbf{r}_2)\\,
            \\phi_\\sigma(\\mathbf{r}_2)\\, d\\mathbf{r}_1\\, d\\mathbf{r}_2

    where :math:`\\phi_\\mu` are contracted Gaussian basis functions.

    The tensor has 8-fold permutational symmetry:

    .. math::

        (\\mu\\nu|\\lambda\\sigma) = (\\nu\\mu|\\lambda\\sigma)
            = (\\mu\\nu|\\sigma\\lambda) = (\\nu\\mu|\\sigma\\lambda)
            = (\\lambda\\sigma|\\mu\\nu) = (\\sigma\\lambda|\\mu\\nu)
            = (\\lambda\\sigma|\\nu\\mu) = (\\sigma\\lambda|\\nu\\mu)

    :param cgtos: List of contracted Gaussian-type orbitals.
    :type cgtos: List[ContractedGaussianTypeOrbital]

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
        The basis functions used to construct the tensor.
    n_basis : int
        Total number of basis functions (counting all angular components).
    tensor : np.ndarray
        The computed ERI tensor of shape ``(n_basis, n_basis, n_basis, n_basis)``.
    matrix : np.ndarray
        Alias for tensor (for compatibility with base class).
    """

    def __init__(self, cgtos: List[ContractedGaussianTypeOrbital]) -> None:
        # Call parent constructor (handles cgtos, n_basis, basis_map, and matrix)
        super().__init__(cgtos)
        # Alias tensor to matrix for semantic clarity
        self.tensor = self.matrix

    # ==================================================================
    # Static methods for Hermite expansion coefficients
    # ==================================================================

    @staticmethod
    def _hermite_expansion_coefficient(
        i: int,
        l1: int,
        l2: int,
        PA: float,
        PB: float,
        gamma: float,
    ) -> float:
        """Compute Hermite expansion coefficient E^{ij}_t.

        The Hermite expansion coefficients relate Cartesian Gaussians to
        Hermite Gaussians via the McMurchie-Davidson scheme:

        .. math::

            (x - A)^{l_1} (x - B)^{l_2} e^{-\\gamma(x-P)^2}
            = \\sum_t E^{l_1 l_2}_t H_t(\\gamma, x - P)

        The recursion is:

        .. math::

            E^{l_1+1, l_2}_t = \\frac{1}{2\\gamma} E^{l_1, l_2}_{t-1}
                              + PA \\cdot E^{l_1, l_2}_t
                              + (t+1) E^{l_1, l_2}_{t+1}

        :param i: Index of the expansion coefficient.
        :type i: int
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

        :returns: Value of the expansion coefficient.
        :rtype: float
        """
        # Valid range check
        if i < 0 or i > l1 + l2:
            return 0.0

        # Base case
        if l1 == 0 and l2 == 0 and i == 0:
            return 1.0

        # Recursion on l1
        if l1 > 0:
            return (
                (1.0 / (2.0 * gamma))
                * TwoElectronRepulsion._hermite_expansion_coefficient(
                    i - 1, l1 - 1, l2, PA, PB, gamma
                )
                + PA
                * TwoElectronRepulsion._hermite_expansion_coefficient(
                    i, l1 - 1, l2, PA, PB, gamma
                )
                + (i + 1)
                * TwoElectronRepulsion._hermite_expansion_coefficient(
                    i + 1, l1 - 1, l2, PA, PB, gamma
                )
            )

        # Recursion on l2 (when l1 == 0)
        return (
            (1.0 / (2.0 * gamma))
            * TwoElectronRepulsion._hermite_expansion_coefficient(
                i - 1, l1, l2 - 1, PA, PB, gamma
            )
            + PB
            * TwoElectronRepulsion._hermite_expansion_coefficient(
                i, l1, l2 - 1, PA, PB, gamma
            )
            + (i + 1)
            * TwoElectronRepulsion._hermite_expansion_coefficient(
                i + 1, l1, l2 - 1, PA, PB, gamma
            )
        )

    # ==================================================================
    # Static methods for Hermite Coulomb integrals
    # ==================================================================

    @staticmethod
    def _hermite_coulomb(
        t: int,
        u: int,
        v: int,
        n: int,
        p: float,
        PC: Tuple[float, float, float],
        RPC_sq: float,
    ) -> float:
        """Compute Hermite Coulomb integral R^n_{tuv} using downward recursion.

        The Hermite Coulomb integrals are auxiliary quantities used in the
        McMurchie-Davidson scheme for Coulomb integrals:

        .. math::

            R^n_{tuv} = (-2p)^n \\cdot \\text{(recursion from Boys function)}

        The recursion starts from:

        .. math::

            R^n_{000} = (-2p)^n F_n(p \\cdot R_{PC}^2)

        :param t: Hermite index in x.
        :type t: int
        :param u: Hermite index in y.
        :type u: int
        :param v: Hermite index in z.
        :type v: int
        :param n: Order of the Boys function.
        :type n: int
        :param p: Reduced exponent for the two-electron integral.
        :type p: float
        :param PC: Vector from product center P to product center Q.
        :type PC: Tuple[float, float, float]
        :param RPC_sq: Squared distance |P - Q|^2.
        :type RPC_sq: float

        :returns: Value of the Hermite Coulomb integral.
        :rtype: float
        """
        # Base case
        if t == 0 and u == 0 and v == 0:
            return ((-2.0 * p) ** n) * boys_function(n, p * RPC_sq)

        # Recursion in x (t > 0)
        if t > 0:
            result = PC[0] * TwoElectronRepulsion._hermite_coulomb(
                t - 1, u, v, n + 1, p, PC, RPC_sq
            )
            if t > 1:
                result += (t - 1) * TwoElectronRepulsion._hermite_coulomb(
                    t - 2, u, v, n + 1, p, PC, RPC_sq
                )
            return result

        # Recursion in y (u > 0)
        if u > 0:
            result = PC[1] * TwoElectronRepulsion._hermite_coulomb(
                t, u - 1, v, n + 1, p, PC, RPC_sq
            )
            if u > 1:
                result += (u - 1) * TwoElectronRepulsion._hermite_coulomb(
                    t, u - 2, v, n + 1, p, PC, RPC_sq
                )
            return result

        # Recursion in z (v > 0)
        if v > 0:
            result = PC[2] * TwoElectronRepulsion._hermite_coulomb(
                t, u, v - 1, n + 1, p, PC, RPC_sq
            )
            if v > 1:
                result += (v - 1) * TwoElectronRepulsion._hermite_coulomb(
                    t, u, v - 2, n + 1, p, PC, RPC_sq
                )
            return result

        return 0.0

    # ==================================================================
    # Primitive and contracted two-electron integrals
    # ==================================================================

    @staticmethod
    def primitive_eri(
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
        gamma: float,
        C: Tuple[float, float, float],
        lx3: int,
        ly3: int,
        lz3: int,
        delta: float,
        D: Tuple[float, float, float],
        lx4: int,
        ly4: int,
        lz4: int,
    ) -> float:
        """Compute two-electron repulsion integral between four primitive Gaussians.

        Evaluates the electron repulsion integral:

        .. math::

            (ab|cd) = \\int\\int g_a(\\mathbf{r}_1)\\, g_b(\\mathbf{r}_1)\\,
                \\frac{1}{|\\mathbf{r}_1 - \\mathbf{r}_2|}\\,
                g_c(\\mathbf{r}_2)\\, g_d(\\mathbf{r}_2)\\,
                d\\mathbf{r}_1\\, d\\mathbf{r}_2

        where :math:`g_a, g_b, g_c, g_d` are normalized Cartesian Gaussian
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
        :param gamma: Exponent of the third primitive.
        :type gamma: float
        :param C: Center of the third primitive (x, y, z) in Bohr.
        :type C: Tuple[float, float, float]
        :param lx3: Angular momentum in x for the third primitive.
        :type lx3: int
        :param ly3: Angular momentum in y for the third primitive.
        :type ly3: int
        :param lz3: Angular momentum in z for the third primitive.
        :type lz3: int
        :param delta: Exponent of the fourth primitive.
        :type delta: float
        :param D: Center of the fourth primitive (x, y, z) in Bohr.
        :type D: Tuple[float, float, float]
        :param lx4: Angular momentum in x for the fourth primitive.
        :type lx4: int
        :param ly4: Angular momentum in y for the fourth primitive.
        :type ly4: int
        :param lz4: Angular momentum in z for the fourth primitive.
        :type lz4: int

        :returns: Two-electron repulsion integral value.
        :rtype: float
        """
        # Combined exponents for each electron
        p = alpha + beta  # Exponent for electron 1
        q = gamma + delta  # Exponent for electron 2

        # Reduced exponent
        rho = p * q / (p + q)

        # Gaussian product centers
        P = TwoElectronRepulsion._gaussian_product_center(alpha, A, beta, B)
        Q = TwoElectronRepulsion._gaussian_product_center(gamma, C, delta, D)

        # Distances from product centers to original centers
        PA = (P[0] - A[0], P[1] - A[1], P[2] - A[2])
        PB = (P[0] - B[0], P[1] - B[1], P[2] - B[2])
        QC = (Q[0] - C[0], Q[1] - C[1], Q[2] - C[2])
        QD = (Q[0] - D[0], Q[1] - D[1], Q[2] - D[2])

        # Vector from P to Q
        PQ = (P[0] - Q[0], P[1] - Q[1], P[2] - Q[2])

        # Squared distances
        AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2
        CD_sq = (C[0] - D[0]) ** 2 + (C[1] - D[1]) ** 2 + (C[2] - D[2]) ** 2
        PQ_sq = PQ[0] ** 2 + PQ[1] ** 2 + PQ[2] ** 2

        # Pre-exponential factors
        K_AB = math.exp(-alpha * beta * AB_sq / p)
        K_CD = math.exp(-gamma * delta * CD_sq / q)

        # Normalization constants
        N1 = normalization_constant(alpha, lx1, ly1, lz1)
        N2 = normalization_constant(beta, lx2, ly2, lz2)
        N3 = normalization_constant(gamma, lx3, ly3, lz3)
        N4 = normalization_constant(delta, lx4, ly4, lz4)

        # Precompute Hermite expansion coefficients (tabular)
        Ex1 = hermite_expansion_coefficients(lx1, lx2, PA[0], PB[0], p)
        Ey1 = hermite_expansion_coefficients(ly1, ly2, PA[1], PB[1], p)
        Ez1 = hermite_expansion_coefficients(lz1, lz2, PA[2], PB[2], p)
        Ex2 = hermite_expansion_coefficients(lx3, lx4, QC[0], QD[0], q)
        Ey2 = hermite_expansion_coefficients(ly3, ly4, QC[1], QD[1], q)
        Ez2 = hermite_expansion_coefficients(lz3, lz4, QC[2], QD[2], q)

        # Precompute Hermite Coulomb integrals (tabular)
        R_table = hermite_coulomb_table(
            lx1 + lx2 + lx3 + lx4,
            ly1 + ly2 + ly3 + ly4,
            lz1 + lz2 + lz3 + lz4,
            rho,
            PQ,
            PQ_sq,
        )

        # Precompute sign-weighted E2 products per (t2, u2, v2)
        # to avoid recomputing them in the inner loops
        integral = 0.0

        for t1 in range(lx1 + lx2 + 1):
            if abs(Ex1[t1]) < 1e-15:
                continue
            for u1 in range(ly1 + ly2 + 1):
                if abs(Ey1[u1]) < 1e-15:
                    continue
                E1_xy = Ex1[t1] * Ey1[u1]
                for v1 in range(lz1 + lz2 + 1):
                    if abs(Ez1[v1]) < 1e-15:
                        continue
                    E1 = E1_xy * Ez1[v1]

                    for t2 in range(lx3 + lx4 + 1):
                        if abs(Ex2[t2]) < 1e-15:
                            continue
                        for u2 in range(ly3 + ly4 + 1):
                            if abs(Ey2[u2]) < 1e-15:
                                continue
                            E2_xy = Ex2[t2] * Ey2[u2]
                            for v2 in range(lz3 + lz4 + 1):
                                if abs(Ez2[v2]) < 1e-15:
                                    continue

                                sign = (-1) ** (t2 + u2 + v2)

                                integral += (
                                    E1
                                    * E2_xy
                                    * Ez2[v2]
                                    * sign
                                    * R_table[t1 + t2, u1 + u2, v1 + v2]
                                )

        # Prefactor: 2π^(5/2) / (p*q*sqrt(p+q))
        prefactor = 2.0 * (math.pi**2.5) / (p * q * math.sqrt(p + q)) * K_AB * K_CD

        return N1 * N2 * N3 * N4 * prefactor * integral

    # ==================================================================
    # Primitive kernel for generic contraction framework
    # ==================================================================

    @staticmethod
    def _primitive_kernel(
        exponents: List[float],
        centers: List[Tuple[float, float, float]],
        angular_momenta: List[Tuple[int, int, int]],
    ) -> float:
        """Primitive ERI kernel for generic n-center contraction.

        This method adapts the primitive_eri computation to the generic
        kernel signature used by :meth:`TwoGaussianIntegral._contract_primitives`.

        :param exponents: List of [alpha, beta, gamma, delta] exponents.
        :type exponents: List[float]
        :param centers: List of [A, B, C, D] atomic centers.
        :type centers: List[Tuple[float, float, float]]
        :param angular_momenta: List of [(lx1,ly1,lz1), ..., (lx4,ly4,lz4)].
        :type angular_momenta: List[Tuple[int, int, int]]

        :returns: Primitive two-electron integral value.
        :rtype: float
        """
        alpha, beta, gamma, delta = exponents
        A, B, C, D = centers
        (lx1, ly1, lz1), (lx2, ly2, lz2), (lx3, ly3, lz3), (lx4, ly4, lz4) = (
            angular_momenta
        )

        return TwoElectronRepulsion.primitive_eri(
            alpha,
            A,
            lx1,
            ly1,
            lz1,
            beta,
            B,
            lx2,
            ly2,
            lz2,
            gamma,
            C,
            lx3,
            ly3,
            lz3,
            delta,
            D,
            lx4,
            ly4,
            lz4,
        )

    # ==================================================================
    # Contracted integral using generic framework
    # ==================================================================

    @staticmethod
    def contracted_eri(
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
        cgto3: ContractedGaussianTypeOrbital,
        lx3: int,
        ly3: int,
        lz3: int,
        cgto4: ContractedGaussianTypeOrbital,
        lx4: int,
        ly4: int,
        lz4: int,
    ) -> float:
        """Compute two-electron repulsion integral between four contracted Gaussians.

        Uses the generic :meth:`TwoGaussianIntegral._contract_primitives` framework
        with the ERI-specific :meth:`_primitive_kernel`.

        Evaluates:

        .. math::

            (\\mu\\nu|\\lambda\\sigma) = \\sum_{p}^{K_\\mu} \\sum_{q}^{K_\\nu}
                \\sum_{r}^{K_\\lambda} \\sum_{s}^{K_\\sigma}
                d_p d_q d_r d_s (g_p g_q | g_r g_s)

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
        :param cgto3: Third contracted Gaussian shell.
        :type cgto3: ContractedGaussianTypeOrbital
        :param lx3: Angular momentum in x for the third function.
        :type lx3: int
        :param ly3: Angular momentum in y for the third function.
        :type ly3: int
        :param lz3: Angular momentum in z for the third function.
        :type lz3: int
        :param cgto4: Fourth contracted Gaussian shell.
        :type cgto4: ContractedGaussianTypeOrbital
        :param lx4: Angular momentum in x for the fourth function.
        :type lx4: int
        :param ly4: Angular momentum in y for the fourth function.
        :type ly4: int
        :param lz4: Angular momentum in z for the fourth function.
        :type lz4: int

        :returns: Two-electron repulsion integral value.
        :rtype: float
        """
        return TwoGaussianIntegral._contract_primitives(
            cgtos=[cgto1, cgto2, cgto3, cgto4],
            angular_momenta=[
                (lx1, ly1, lz1),
                (lx2, ly2, lz2),
                (lx3, ly3, lz3),
                (lx4, ly4, lz4),
            ],
            primitive_kernel=TwoElectronRepulsion._primitive_kernel,
        )

    # ==================================================================
    # Abstract method implementation (required by TwoGaussianIntegral)
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
        """Not used for two-electron integrals.

        This method is required by the abstract base class but is not used
        for four-center integrals. The 4-center computation uses
        :meth:`_compute_element_4center` instead.

        :raises NotImplementedError: Always, as this method should not be called.
        """
        raise NotImplementedError(
            "Two-electron integrals use _compute_element_4center, not _compute_element."
        )

    def _compute_element_4center(
        self,
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
        cgto3: ContractedGaussianTypeOrbital,
        lx3: int,
        ly3: int,
        lz3: int,
        cgto4: ContractedGaussianTypeOrbital,
        lx4: int,
        ly4: int,
        lz4: int,
    ) -> float:
        """Compute a single 4-center ERI element using generic contraction.

        Uses :meth:`TwoGaussianIntegral._contract_primitives` with :meth:`_primitive_kernel`.

        :returns: Contracted two-electron integral value (μν|λσ).
        :rtype: float
        """
        return self._contract_primitives(
            cgtos=[cgto1, cgto2, cgto3, cgto4],
            angular_momenta=[
                (lx1, ly1, lz1),
                (lx2, ly2, lz2),
                (lx3, ly3, lz3),
                (lx4, ly4, lz4),
            ],
            primitive_kernel=self._primitive_kernel,
        )

    # ==================================================================
    # Tensor computation (overrides parent's 2D matrix computation)
    # ==================================================================

    def _compute_matrix(self) -> np.ndarray:
        """Compute the full ERI tensor.

        Uses 8-fold permutational symmetry to reduce computation:
        only computes unique elements and fills in symmetric partners.
        Applies Schwarz screening to skip negligible integrals.

        :returns: ERI tensor of shape ``(n_basis, n_basis, n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        tensor = np.zeros((self.n_basis, self.n_basis, self.n_basis, self.n_basis))

        # --- Schwarz screening: precompute (μν|μν) bounds ----
        schwarz_threshold = 1e-12
        schwarz = np.zeros((self.n_basis, self.n_basis))
        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]
            for nu in range(mu + 1):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]
                diag = self._compute_element_4center(
                    cgto1,
                    lx1,
                    ly1,
                    lz1,
                    cgto2,
                    lx2,
                    ly2,
                    lz2,
                    cgto1,
                    lx1,
                    ly1,
                    lz1,
                    cgto2,
                    lx2,
                    ly2,
                    lz2,
                )
                val = math.sqrt(abs(diag))
                schwarz[mu, nu] = val
                schwarz[nu, mu] = val

        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]

            for nu in range(mu + 1):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]

                # Compound index for (μν) pair
                mn = mu * (mu + 1) // 2 + nu
                Q_mn = schwarz[mu, nu]

                for lam in range(self.n_basis):
                    shell_lam, lx3, ly3, lz3 = self._basis_map[lam]
                    cgto3 = self.cgtos[shell_lam]

                    for sig in range(lam + 1):
                        # Compound index for (λσ) pair
                        ls = lam * (lam + 1) // 2 + sig

                        # Only compute if (μν) >= (λσ) in compound index
                        if mn < ls:
                            continue

                        # Schwarz screening
                        if Q_mn * schwarz[lam, sig] < schwarz_threshold:
                            continue

                        shell_sig, lx4, ly4, lz4 = self._basis_map[sig]
                        cgto4 = self.cgtos[shell_sig]

                        element = self._compute_element_4center(
                            cgto1,
                            lx1,
                            ly1,
                            lz1,
                            cgto2,
                            lx2,
                            ly2,
                            lz2,
                            cgto3,
                            lx3,
                            ly3,
                            lz3,
                            cgto4,
                            lx4,
                            ly4,
                            lz4,
                        )

                        # Fill all 8 permutations
                        tensor[mu, nu, lam, sig] = element
                        tensor[nu, mu, lam, sig] = element
                        tensor[mu, nu, sig, lam] = element
                        tensor[nu, mu, sig, lam] = element
                        tensor[lam, sig, mu, nu] = element
                        tensor[sig, lam, mu, nu] = element
                        tensor[lam, sig, nu, mu] = element
                        tensor[sig, lam, nu, mu] = element

        return tensor

    # ==================================================================
    # Access methods
    # ==================================================================

    def __getitem__(self, key: Tuple[int, int, int, int]) -> float:
        """Access tensor element by index.

        :param key: Tuple of (μ, ν, λ, σ) indices.
        :type key: Tuple[int, int, int, int]

        :returns: Tensor element (μν|λσ).
        :rtype: float
        """
        return self.tensor[key]

    def __repr__(self) -> str:
        """String representation showing class name and basis size."""
        return f"TwoElectronRepulsion(n_basis={self.n_basis})"

    def __array__(self) -> np.ndarray:
        """Return the tensor as a NumPy array.

        Enables ``np.array(eri)`` conversion.

        :returns: The ERI tensor.
        :rtype: np.ndarray
        """
        return self.tensor