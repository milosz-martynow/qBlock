"""Nuclear attraction matrix computation for Gaussian basis functions.

This module computes the nuclear attraction matrix :math:`V` between contracted
Gaussian-type orbitals (CGTOs):

.. math::

    V_{\\mu\\nu} = \\sum_C \\int \\phi_\\mu(\\mathbf{r})\\,
        \\left(-\\frac{Z_C}{|\\mathbf{r} - \\mathbf{R}_C|}\\right)\\,
        \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

where :math:`\\phi_\\mu` is a contracted Gaussian basis function,
:math:`Z_C` is the nuclear charge at center :math:`\\mathbf{R}_C`.

The Coulomb integral over Gaussians is evaluated using the Boys function
:math:`F_n(x)` and the McMurchie-Davidson recurrence relations for
Hermite Coulomb integrals.

Classes
-------
NuclearAttraction
    Computes and stores the full nuclear attraction matrix for a molecular basis.
"""

import math
from typing import List, Tuple

import numpy as np

from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.integrals.overlap import Overlap
from q_block.theory.utils import normalization_constant, boys_function


# ======================================================================
# Main nuclear attraction matrix class
# ======================================================================


class NuclearAttraction(Overlap):
    """Computes and stores the nuclear attraction matrix V for a molecular basis.

    Inherits from :class:`Overlap` to reuse Gaussian product center
    computation and basis index mapping methods.

    The nuclear attraction matrix elements are:

    .. math::

        V_{\\mu\\nu} = \\sum_C \\int \\phi_\\mu(\\mathbf{r})\\,
            \\left(-\\frac{Z_C}{|\\mathbf{r} - \\mathbf{R}_C|}\\right)\\,
            \\phi_\\nu(\\mathbf{r})\\, d\\mathbf{r}

    where :math:`\\phi_\\mu` are contracted Gaussian basis functions and
    the sum runs over all nuclei :math:`C`.

    The matrix is symmetric: :math:`V_{\\mu\\nu} = V_{\\nu\\mu}`.

    :param cgtos: List of contracted Gaussian-type orbitals.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: List of (charge, position) tuples for each nucleus,
        where charge is the atomic number and position is (x, y, z) in Bohr.
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
        The basis functions used to construct the matrix.
    nuclei : List[Tuple[int, Tuple[float, float, float]]]
        Nuclear charges and positions.
    n_basis : int
        Total number of basis functions (counting all angular components).
    matrix : np.ndarray
        The computed nuclear attraction matrix of shape ``(n_basis, n_basis)``.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
    ) -> None:
        if not cgtos:
            raise ValueError("Cannot build nuclear attraction matrix with empty basis set.")
        if not nuclei:
            raise ValueError("Cannot build nuclear attraction matrix without nuclei.")

        self.cgtos: List[ContractedGaussianTypeOrbital] = cgtos
        self.nuclei: List[Tuple[int, Tuple[float, float, float]]] = nuclei

        # Count total basis functions (each shell contributes 2l+1 functions)
        self.n_basis: int = sum(cgto.n_functions for cgto in cgtos)

        # Build the basis function index mapping (inherited from Overlap)
        self._build_basis_index_map()

        # Compute the nuclear attraction matrix
        self.matrix: np.ndarray = self._compute_nuclear_attraction_matrix()

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
        :param p: Sum of exponents (alpha + beta).
        :type p: float
        :param PC: Vector from product center P to nucleus C.
        :type PC: Tuple[float, float, float]
        :param RPC_sq: Squared distance |P - C|^2.
        :type RPC_sq: float

        :returns: Value of the Hermite Coulomb integral.
        :rtype: float
        """
        # Base case
        if t == 0 and u == 0 and v == 0:
            return ((-2.0 * p) ** n) * boys_function(n, p * RPC_sq)

        # Recursion in x (t > 0)
        if t > 0:
            result = PC[0] * NuclearAttraction._hermite_coulomb(
                t - 1, u, v, n + 1, p, PC, RPC_sq
            )
            if t > 1:
                result += (t - 1) * NuclearAttraction._hermite_coulomb(
                    t - 2, u, v, n + 1, p, PC, RPC_sq
                )
            return result

        # Recursion in y (u > 0)
        if u > 0:
            result = PC[1] * NuclearAttraction._hermite_coulomb(
                t, u - 1, v, n + 1, p, PC, RPC_sq
            )
            if u > 1:
                result += (u - 1) * NuclearAttraction._hermite_coulomb(
                    t, u - 2, v, n + 1, p, PC, RPC_sq
                )
            return result

        # Recursion in z (v > 0)
        if v > 0:
            result = PC[2] * NuclearAttraction._hermite_coulomb(
                t, u, v - 1, n + 1, p, PC, RPC_sq
            )
            if v > 1:
                result += (v - 1) * NuclearAttraction._hermite_coulomb(
                    t, u, v - 2, n + 1, p, PC, RPC_sq
                )
            return result

        return 0.0

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
                * NuclearAttraction._hermite_expansion_coefficient(
                    i - 1, l1 - 1, l2, PA, PB, gamma
                )
                + PA
                * NuclearAttraction._hermite_expansion_coefficient(
                    i, l1 - 1, l2, PA, PB, gamma
                )
                + (i + 1)
                * NuclearAttraction._hermite_expansion_coefficient(
                    i + 1, l1 - 1, l2, PA, PB, gamma
                )
            )

        # Recursion on l2 (when l1 == 0)
        return (
            (1.0 / (2.0 * gamma))
            * NuclearAttraction._hermite_expansion_coefficient(
                i - 1, l1, l2 - 1, PA, PB, gamma
            )
            + PB
            * NuclearAttraction._hermite_expansion_coefficient(
                i, l1, l2 - 1, PA, PB, gamma
            )
            + (i + 1)
            * NuclearAttraction._hermite_expansion_coefficient(
                i + 1, l1, l2 - 1, PA, PB, gamma
            )
        )

    @staticmethod
    def primitive_nuclear_attraction(
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
        C: Tuple[float, float, float],
        Z: int,
    ) -> float:
        """Compute nuclear attraction integral between two primitive Gaussians.

        Evaluates:

        .. math::

            V_C = -Z_C \\int g_a(\\mathbf{r})\\,
                \\frac{1}{|\\mathbf{r} - \\mathbf{R}_C|}\\,
                g_b(\\mathbf{r})\\, d\\mathbf{r}

        where :math:`g_a` and :math:`g_b` are normalized Cartesian Gaussian
        primitives and :math:`Z_C` is the nuclear charge at :math:`\\mathbf{R}_C`.

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
        :param C: Position of the nucleus (x, y, z) in Bohr.
        :type C: Tuple[float, float, float]
        :param Z: Nuclear charge (atomic number).
        :type Z: int

        :returns: Nuclear attraction integral value.
        :rtype: float
        """
        gamma = alpha + beta

        # Gaussian product center (inherited from Overlap)
        P = Overlap._gaussian_product_center(alpha, A, beta, B)

        # Distances from product center
        PA = (P[0] - A[0], P[1] - A[1], P[2] - A[2])
        PB = (P[0] - B[0], P[1] - B[1], P[2] - B[2])
        PC = (P[0] - C[0], P[1] - C[1], P[2] - C[2])

        # Squared distances
        AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2
        PC_sq = PC[0] ** 2 + PC[1] ** 2 + PC[2] ** 2

        # Pre-exponential factor
        pre_factor = math.exp(-alpha * beta * AB_sq / gamma)

        # Normalization constants
        N1 = normalization_constant(alpha, lx1, ly1, lz1)
        N2 = normalization_constant(beta, lx2, ly2, lz2)

        # Compute the Coulomb integral using Hermite expansion
        integral = 0.0
        for t in range(lx1 + lx2 + 1):
            E_x = NuclearAttraction._hermite_expansion_coefficient(
                t, lx1, lx2, PA[0], PB[0], gamma
            )
            if abs(E_x) < 1e-15:
                continue

            for u in range(ly1 + ly2 + 1):
                E_y = NuclearAttraction._hermite_expansion_coefficient(
                    u, ly1, ly2, PA[1], PB[1], gamma
                )
                if abs(E_y) < 1e-15:
                    continue

                for v in range(lz1 + lz2 + 1):
                    E_z = NuclearAttraction._hermite_expansion_coefficient(
                        v, lz1, lz2, PA[2], PB[2], gamma
                    )
                    if abs(E_z) < 1e-15:
                        continue

                    R_tuv = NuclearAttraction._hermite_coulomb(
                        t, u, v, 0, gamma, PC, PC_sq
                    )
                    integral += E_x * E_y * E_z * R_tuv

        # Factor of 2π/γ from the Coulomb integral formula
        integral *= (2.0 * math.pi / gamma) * pre_factor

        return -Z * N1 * N2 * integral

    @staticmethod
    def contracted_nuclear_attraction(
        cgto1: ContractedGaussianTypeOrbital,
        lx1: int,
        ly1: int,
        lz1: int,
        cgto2: ContractedGaussianTypeOrbital,
        lx2: int,
        ly2: int,
        lz2: int,
        C: Tuple[float, float, float],
        Z: int,
    ) -> float:
        """Compute nuclear attraction integral between two contracted Gaussians.

        Evaluates:

        .. math::

            V^C_{\\mu\\nu} = -Z_C \\sum_{p}^{K_\\mu} \\sum_{q}^{K_\\nu}
                d_p d_q \\langle g_p | \\frac{1}{|\\mathbf{r} - \\mathbf{R}_C|} | g_q \\rangle

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
        :param C: Position of the nucleus (x, y, z) in Bohr.
        :type C: Tuple[float, float, float]
        :param Z: Nuclear charge (atomic number).
        :type Z: int

        :returns: Nuclear attraction integral value.
        :rtype: float
        """
        A = (cgto1.center.x, cgto1.center.y, cgto1.center.z)
        B = (cgto2.center.x, cgto2.center.y, cgto2.center.z)

        attraction = 0.0

        for p in range(cgto1.n_primitives):
            alpha = cgto1.exponents[p]
            d_p = cgto1.contractions[p]

            for q in range(cgto2.n_primitives):
                beta = cgto2.exponents[q]
                d_q = cgto2.contractions[q]

                prim_attr = NuclearAttraction.primitive_nuclear_attraction(
                    alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2, C, Z
                )
                attraction += d_p * d_q * prim_attr

        return attraction

    # ==================================================================
    # Instance methods
    # ==================================================================

    def _compute_nuclear_attraction_matrix(self) -> np.ndarray:
        """Compute the full nuclear attraction matrix.

        Uses symmetry: only computes upper triangle and mirrors to lower.
        Sums contributions from all nuclei.

        :returns: Nuclear attraction matrix of shape ``(n_basis, n_basis)``.
        :rtype: np.ndarray
        """
        V = np.zeros((self.n_basis, self.n_basis))

        for mu in range(self.n_basis):
            shell_mu, lx1, ly1, lz1 = self._basis_map[mu]
            cgto1 = self.cgtos[shell_mu]

            for nu in range(mu, self.n_basis):
                shell_nu, lx2, ly2, lz2 = self._basis_map[nu]
                cgto2 = self.cgtos[shell_nu]

                # Sum over all nuclei
                V_mn = 0.0
                for Z, C in self.nuclei:
                    V_mn += self.contracted_nuclear_attraction(
                        cgto1, lx1, ly1, lz1, cgto2, lx2, ly2, lz2, C, Z
                    )

                V[mu, nu] = V_mn
                V[nu, mu] = V_mn  # Symmetry

        return V

    def __repr__(self) -> str:
        return f"NuclearAttraction(n_basis={self.n_basis}, n_nuclei={len(self.nuclei)})"

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Access matrix element by index.

        :param key: Tuple of (row, column) indices.
        :type key: Tuple[int, int]

        :returns: Matrix element V[row, col].
        :rtype: float
        """
        return self.matrix[key]
