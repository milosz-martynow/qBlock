"""Contracted Gaussian-Type Orbital (CGTO) basis functions.

This module provides classes that represent the mathematical basis
functions used in Hartree-Fock integral computation:

.. math::

    \\phi_\\mu(\\mathbf{r})
        = \\sum_{p=1}^{K_\\mu} d_{p\\mu}\\,
          N_{p\\mu}\\,
          (x - R_x)^{l_x}\\,
          (y - R_y)^{l_y}\\,
          (z - R_z)^{l_z}\\,
          e^{-\\alpha_{p\\mu} |\\mathbf{r} - \\mathbf{R}_\\mu|^2}

These are **not** electron orbitals — they are fixed building blocks
(determined entirely by the basis set and molecular geometry) from which
molecular orbitals are constructed via:

.. math::

    \\psi_i(\\mathbf{r}) = \\sum_{\\mu=1}^{n_{basis}} C_{\\mu i}\\, \\phi_\\mu(\\mathbf{r})

A single contracted shell with angular momentum :math:`l` shares one set
of exponents and contraction coefficients across all :math:`2l+1`
angular components.  The magnetic quantum number is therefore **not**
stored on the CGTO — it is handled during integral evaluation.

Classes
-------
ContractedGaussianTypeOrbital
    A single CGTO (one contracted shell).

CGTOBasis
    A flat collection of all CGTO shells for a
    :class:`~q_block.systems.molecule.Molecule`, ready for integral
    computation.
"""

import math
from typing import List, Optional, Tuple

from q_block.io.coordinates import CartesianCoordinates


# ======================================================================
# Single contracted shell
# ======================================================================


class ContractedGaussianTypeOrbital:
    r"""A single contracted Gaussian-type orbital (CGTO) basis function.

    Represents one contracted shell centred on an atom:

    .. math::

        \phi(\mathbf{r})
            = \sum_{p=1}^{K} d_p\,
              g_p(\alpha_p,\, \mathbf{r} - \mathbf{R})

    where each primitive :math:`g_p` is a Cartesian Gaussian.

    A shell with angular momentum :math:`l` produces :math:`2l + 1`
    individual basis functions (one per Cartesian/spherical-harmonic
    component).  The contraction parameters are identical for every
    component, so the magnetic quantum number is **not** part of this
    object — it is resolved during integral evaluation.

    :param center: Position :math:`\mathbf{R}` of the function centre
        (in Bohr).
    :type center: CartesianCoordinates
    :param l: Angular momentum quantum number
        (:math:`l = 0` s, :math:`l = 1` p, :math:`l = 2` d, …).
    :type l: int
    :param exponents: Gaussian exponents
        :math:`(\alpha_1, \ldots, \alpha_K)`.
    :type exponents: List[float]
    :param contractions: Contraction coefficients
        :math:`(d_1, \ldots, d_K)`.
    :type contractions: List[float]
    :param atom_index: Index of the atom in
        :attr:`Molecule.atoms` that this function is centred on.
        ``None`` when the CGTO is built at the :class:`Atom` level
        (before a molecule context exists).
    :type atom_index: Optional[int]

    Attributes
    ----------
    center : CartesianCoordinates
        Function centre in Bohr.
    l : int
        Angular momentum quantum number.
    exponents : List[float]
        Primitive Gaussian exponents :math:`\alpha_p`.
    contractions : List[float]
        Contraction coefficients :math:`d_p`.
    atom_index : Optional[int]
        Owning atom index in the parent Molecule (``None`` at atom level).
    n_primitives : int
        Number of primitive Gaussians in this contraction.
    n_functions : int
        Number of basis functions this shell contributes (:math:`2l + 1`).
    """

    def __init__(
        self,
        center: CartesianCoordinates,
        l: int,
        exponents: List[float],
        contractions: List[float],
        atom_index: Optional[int] = None,
    ) -> None:
        if len(exponents) != len(contractions):
            raise ValueError(
                f"exponents and contractions must have the same length; "
                f"got {len(exponents)} vs {len(contractions)}."
            )
        if not exponents:
            raise ValueError("A CGTO must have at least one primitive.")

        self.center: CartesianCoordinates = center
        self.l: int = l
        self.exponents: List[float] = list(exponents)
        self.contractions: List[float] = list(contractions)
        self.atom_index: Optional[int] = atom_index
        self.n_primitives: int = len(exponents)
        self.n_functions: int = 2 * l + 1

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lbl = "spdfghiklm"[self.l] if self.l < 10 else f"l{self.l}"
        return (
            f"CGTO(atom={self.atom_index}, {lbl}, "
            f"K={self.n_primitives}, R={self.center})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ContractedGaussianTypeOrbital):
            return NotImplemented
        return (
            self.center == other.center
            and self.l == other.l
            and self.exponents == other.exponents
            and self.contractions == other.contractions
            and self.atom_index == other.atom_index
        )

    # ------------------------------------------------------------------
    # Evaluation methods
    # ------------------------------------------------------------------

    def primitive_gaussian(
        self,
        r: CartesianCoordinates,
        primitive_index: int,
        lx: int,
        ly: int,
        lz: int,
    ) -> float:
        r"""Evaluate a single primitive Cartesian Gaussian at position r.

        Computes:

        .. math::

            g_p(\mathbf{r}) = N\,
                (x - R_x)^{l_x}\,
                (y - R_y)^{l_y}\,
                (z - R_z)^{l_z}\,
                e^{-\alpha_p |\mathbf{r} - \mathbf{R}|^2}

        where :math:`N` is the normalisation constant for a Cartesian
        Gaussian.

        :param r: Point at which to evaluate the primitive.
        :type r: CartesianCoordinates
        :param primitive_index: Index (0-based) of the primitive to
            evaluate.
        :type primitive_index: int
        :param lx: Cartesian angular momentum in :math:`x`.
        :type lx: int
        :param ly: Cartesian angular momentum in :math:`y`.
        :type ly: int
        :param lz: Cartesian angular momentum in :math:`z`.
        :type lz: int

        :returns: Value of the primitive Gaussian at :math:`\mathbf{r}`.
        :rtype: float

        :raises IndexError: If ``primitive_index`` is out of range.
        :raises ValueError: If ``lx + ly + lz != self.l``.
        """
        if primitive_index < 0 or primitive_index >= self.n_primitives:
            raise IndexError(
                f"primitive_index {primitive_index} out of range "
                f"[0, {self.n_primitives})."
            )
        if lx + ly + lz != self.l:
            raise ValueError(
                f"lx + ly + lz = {lx + ly + lz} must equal l = {self.l}."
            )

        alpha = self.exponents[primitive_index]

        dx = r.x - self.center.x
        dy = r.y - self.center.y
        dz = r.z - self.center.z
        r_sq = dx * dx + dy * dy + dz * dz

        # Normalisation constant for Cartesian Gaussian
        norm = self._normalisation_constant(alpha, lx, ly, lz)

        angular = (dx ** lx) * (dy ** ly) * (dz ** lz)
        radial = math.exp(-alpha * r_sq)

        return norm * angular * radial

    def basis_function(
        self,
        r: CartesianCoordinates,
        lx: int,
        ly: int,
        lz: int,
    ) -> float:
        r"""Evaluate the contracted basis function at position r.

        Computes:

        .. math::

            \phi(\mathbf{r}) = \sum_{p=1}^{K} d_p\, g_p(\mathbf{r})

        for a specific Cartesian angular component :math:`(l_x, l_y, l_z)`.

        :param r: Point at which to evaluate the basis function.
        :type r: CartesianCoordinates
        :param lx: Cartesian angular momentum in :math:`x`.
        :type lx: int
        :param ly: Cartesian angular momentum in :math:`y`.
        :type ly: int
        :param lz: Cartesian angular momentum in :math:`z`.
        :type lz: int

        :returns: Value of the contracted Gaussian at :math:`\mathbf{r}`.
        :rtype: float

        :raises ValueError: If ``lx + ly + lz != self.l``.
        """
        if lx + ly + lz != self.l:
            raise ValueError(
                f"lx + ly + lz = {lx + ly + lz} must equal l = {self.l}."
            )

        value = 0.0
        for p in range(self.n_primitives):
            coeff = self.contractions[p]
            value += coeff * self.primitive_gaussian(r, p, lx, ly, lz)

        return value

    @staticmethod
    def _normalisation_constant(
        alpha: float,
        lx: int,
        ly: int,
        lz: int,
    ) -> float:
        r"""Compute normalisation constant for a Cartesian Gaussian primitive.

        .. math::

            N = \left( \frac{2\alpha}{\pi} \right)^{3/4}
                \left( \frac{(4\alpha)^{l_x + l_y + l_z}}
                             {(2l_x - 1)!! (2l_y - 1)!! (2l_z - 1)!!}
                \right)^{1/2}

        :param alpha: Gaussian exponent.
        :type alpha: float
        :param lx: Angular momentum in x.
        :type lx: int
        :param ly: Angular momentum in y.
        :type ly: int
        :param lz: Angular momentum in z.
        :type lz: int

        :returns: Normalisation constant.
        :rtype: float
        """
        l_total = lx + ly + lz

        prefactor = (2.0 * alpha / math.pi) ** 0.75
        numerator = (4.0 * alpha) ** l_total
        denominator = (
            ContractedGaussianTypeOrbital._double_factorial(2 * lx - 1)
            * ContractedGaussianTypeOrbital._double_factorial(2 * ly - 1)
            * ContractedGaussianTypeOrbital._double_factorial(2 * lz - 1)
        )

        return prefactor * math.sqrt(numerator / denominator)

    @staticmethod
    def _double_factorial(n: int) -> int:
        """Compute double factorial n!!.

        By convention, ``(-1)!! = 1`` and ``0!! = 1``.

        :param n: Non-negative integer (or -1).
        :type n: int
        :returns: n!!
        :rtype: int
        """
        if n <= 0:
            return 1
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result
