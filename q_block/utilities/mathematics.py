"""Common mathematical utilities for quantum-chemistry computations.

This module provides shared mathematical functions used across
the theory package, particularly for Gaussian basis function
computations.

Functions
---------
double_factorial
    Computes the double factorial n!! with quantum chemistry convention.
normalization_constant
    Computes the normalization constant for Cartesian Gaussian primitives.
get_cartesian_components
    Returns all Cartesian angular momentum components for given l.
boys_function
    Computes the Boys function F_n(x) for Coulomb integrals.
boys_function_array
    Computes F_0(x) through F_{n_max}(x) with one hyp1f1 call + downward recursion.
hermite_expansion_coefficients
    Tabular bottom-up computation of McMurchie-Davidson E coefficients.
hermite_coulomb_table
    Tabular bottom-up computation of Hermite Coulomb integrals R^0_{tuv}.
"""

import functools
import math

import numpy as np
from scipy.special import factorial2 as _scipy_factorial2
from scipy.special import hyp1f1 as _scipy_hyp1f1


@functools.lru_cache(maxsize=None)
def double_factorial(n: int) -> int:
    """Compute double factorial n!! with quantum chemistry convention.

    Uses scipy's ``factorial2`` internally but handles the special case
    where ``n <= 0``. By quantum chemistry convention:

    - ``(-1)!! = 1``
    - ``0!! = 1``

    scipy's ``factorial2(-1)`` returns 0, which would cause division by
    zero in normalization computations.

    :param n: Integer argument.
    :type n: int

    :returns: Double factorial n!!
    :rtype: int

    Examples
    --------
    >>> double_factorial(-1)
    1
    >>> double_factorial(0)
    1
    >>> double_factorial(5)
    15
    """
    if n <= 0:
        return 1
    return int(_scipy_factorial2(n, exact=True))


@functools.lru_cache(maxsize=None)
def normalization_constant(alpha: float, lx: int, ly: int, lz: int) -> float:
    r"""Compute normalization constant for a Cartesian Gaussian primitive.

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

    :returns: Normalization constant.
    :rtype: float
    """
    l_total = lx + ly + lz
    prefactor = (2.0 * alpha / math.pi) ** 0.75
    numerator = (4.0 * alpha) ** l_total

    denominator = (
        double_factorial(2 * lx - 1)
        * double_factorial(2 * ly - 1)
        * double_factorial(2 * lz - 1)
    )

    return prefactor * math.sqrt(numerator / denominator)


def get_cartesian_components(l: int) -> list:
    """Get all Cartesian angular momentum components for a given l.

    For angular momentum :math:`l`, generates all :math:`(l_x, l_y, l_z)`
    tuples such that :math:`l_x + l_y + l_z = l`.

    The ordering follows the convention used in most quantum chemistry
    codes (descending in :math:`l_x`, then :math:`l_y`):

    - l=0 (s): [(0,0,0)]
    - l=1 (p): [(1,0,0), (0,1,0), (0,0,1)]
    - l=2 (d): [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]

    :param l: Total angular momentum quantum number.
    :type l: int

    :returns: List of (lx, ly, lz) tuples.
    :rtype: list
    """
    components = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            components.append((lx, ly, lz))
    return components


def boys_function(n: int, x: float) -> float:
    r"""Compute the Boys function F_n(x).

    The Boys function is defined as:

    .. math::

        F_n(x) = \int_0^1 t^{2n} e^{-x t^2} dt

    This function is essential for evaluating Coulomb integrals
    (nuclear attraction and electron repulsion) over Gaussian basis
    functions.

    For small x, the Boys function is computed using the confluent
    hypergeometric function:

    .. math::

        F_n(x) = \frac{1}{2n+1} \cdot {}_1F_1\left(n + \frac{1}{2}; n + \frac{3}{2}; -x\right)

    For large x, an asymptotic expansion is used:

    .. math::

        F_n(x) \approx \frac{(2n-1)!!}{2^{n+1}} \sqrt{\frac{\pi}{x^{2n+1}}}

    :param n: Order of the Boys function (non-negative integer).
    :type n: int
    :param x: Argument of the Boys function (non-negative real).
    :type x: float

    :returns: Value of F_n(x).
    :rtype: float

    Examples
    --------
    >>> boys_function(0, 0.0)
    1.0
    >>> boys_function(0, 1.0)  # doctest: +ELLIPSIS
    0.746824...
    """
    if x < 1e-10:
        # Taylor expansion for small x: F_n(x) ≈ 1/(2n+1) - x/(2n+3) + ...
        return 1.0 / (2 * n + 1)

    # Use confluent hypergeometric function for general case
    # F_n(x) = (1/(2n+1)) * 1F1(n+1/2, n+3/2, -x)
    return _scipy_hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)


def boys_function_array(n_max: int, x: float) -> list:
    r"""Compute Boys function values F_0(x) through F_{n_max}(x).

    Uses a single evaluation of F_{n_max}(x) via the confluent
    hypergeometric function, then applies the numerically stable
    downward recursion:

    .. math::

        F_n(x) = \frac{2x\,F_{n+1}(x) + e^{-x}}{2n + 1}

    :param n_max: Maximum order.
    :type n_max: int
    :param x: Argument (non-negative).
    :type x: float
    :returns: List ``[F_0(x), F_1(x), ..., F_{n_max}(x)]``.
    :rtype: list
    """
    if n_max < 0:
        return []

    result = [0.0] * (n_max + 1)

    if x < 1e-10:
        for n in range(n_max + 1):
            result[n] = 1.0 / (2 * n + 1)
        return result

    # Compute F_{n_max} directly
    result[n_max] = _scipy_hyp1f1(n_max + 0.5, n_max + 1.5, -x) / (2 * n_max + 1)

    # Downward recursion
    exp_neg_x = math.exp(-x)
    two_x = 2.0 * x
    for n in range(n_max - 1, -1, -1):
        result[n] = (two_x * result[n + 1] + exp_neg_x) / (2 * n + 1)

    return result


def hermite_expansion_coefficients(
    l1: int, l2: int, PA: float, PB: float, gamma: float
) -> list:
    r"""Compute all Hermite expansion coefficients E_t for t = 0 .. l1+l2.

    Replaces the recursive McMurchie-Davidson evaluation with a
    bottom-up tabulation that eliminates redundant sub-problem
    recomputation.

    The recursion builds up in two phases:

    1. Increment l2 from 0 to its target (with l1 = 0).
    2. Increment l1 from 0 to its target (with l2 fixed).

    :param l1: Angular momentum on center A.
    :type l1: int
    :param l2: Angular momentum on center B.
    :type l2: int
    :param PA: Component of P − A along one Cartesian axis.
    :type PA: float
    :param PB: Component of P − B along one Cartesian axis.
    :type PB: float
    :param gamma: Sum of exponents (alpha + beta).
    :type gamma: float
    :returns: List of length ``l1 + l2 + 1`` with ``E_t`` values.
    :rtype: list
    """
    t_max = l1 + l2
    inv_2gamma = 0.5 / gamma

    # prev[t] holds the current coefficients; extra slot for t+1 access
    prev = [0.0] * (t_max + 2)
    prev[0] = 1.0  # E(0, 0, 0) = 1

    # Phase 1: build E(0, b, t) for b = 1 .. l2
    for b in range(l2):
        curr = [0.0] * (t_max + 2)
        for t in range(b + 2):
            val = PB * prev[t]
            if t > 0:
                val += inv_2gamma * prev[t - 1]
            val += (t + 1) * prev[t + 1]
            curr[t] = val
        prev = curr

    # Phase 2: build E(a, l2, t) for a = 1 .. l1
    for a in range(l1):
        curr = [0.0] * (t_max + 2)
        for t in range(a + l2 + 2):
            val = PA * prev[t]
            if t > 0:
                val += inv_2gamma * prev[t - 1]
            val += (t + 1) * prev[t + 1]
            curr[t] = val
        prev = curr

    return prev[: t_max + 1]


def hermite_coulomb_table(
    t_max: int,
    u_max: int,
    v_max: int,
    p: float,
    PC: tuple,
    RPC_sq: float,
) -> np.ndarray:
    r"""Precompute Hermite Coulomb integrals R^0_{tuv}.

    Builds a 3D table ``R[t, u, v]`` using bottom-up recursion,
    replacing the naive recursive evaluation that suffers from
    exponential recomputation of overlapping sub-problems.

    Internally calls :func:`boys_function_array` once for
    ``F_0 \ldots F_{N}`` and then propagates through the three
    Cartesian directions.

    :param t_max: Maximum Hermite index in x.
    :type t_max: int
    :param u_max: Maximum Hermite index in y.
    :type u_max: int
    :param v_max: Maximum Hermite index in z.
    :type v_max: int
    :param p: Exponent parameter (gamma for 2-center, rho for
        4-center integrals).
    :type p: float
    :param PC: Vector from product centre to target centre
        (or P−Q for ERIs).
    :type PC: tuple
    :param RPC_sq: Squared distance ``|P − C|^2`` (or ``|P − Q|^2``).
    :type RPC_sq: float
    :returns: Array of shape ``(t_max+1, u_max+1, v_max+1)``.
    :rtype: np.ndarray
    """
    N = t_max + u_max + v_max

    boys_vals = boys_function_array(N, p * RPC_sq)

    # 4D work array; last axis is the auxiliary order n
    R = np.zeros((t_max + 1, u_max + 1, v_max + 1, N + 2))

    # Base case: R(0,0,0,n) = (-2p)^n * F_n(arg)
    minus_2p = -2.0 * p
    factor = 1.0
    for n in range(N + 1):
        R[0, 0, 0, n] = factor * boys_vals[n]
        factor *= minus_2p

    # Build up in v (t=0, u=0)
    for v in range(1, v_max + 1):
        n_end = N - v
        for n in range(n_end + 1):
            R[0, 0, v, n] = PC[2] * R[0, 0, v - 1, n + 1]
            if v > 1:
                R[0, 0, v, n] += (v - 1) * R[0, 0, v - 2, n + 1]

    # Build up in u (t=0)
    for u in range(1, u_max + 1):
        for v in range(v_max + 1):
            n_end = N - u - v
            for n in range(n_end + 1):
                R[0, u, v, n] = PC[1] * R[0, u - 1, v, n + 1]
                if u > 1:
                    R[0, u, v, n] += (u - 1) * R[0, u - 2, v, n + 1]

    # Build up in t
    for t in range(1, t_max + 1):
        for u in range(u_max + 1):
            for v in range(v_max + 1):
                n_end = N - t - u - v
                for n in range(n_end + 1):
                    R[t, u, v, n] = PC[0] * R[t - 1, u, v, n + 1]
                    if t > 1:
                        R[t, u, v, n] += (t - 1) * R[t - 2, u, v, n + 1]

    return R[:, :, :, 0]
