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
"""

import math

import numpy as np
from scipy.special import factorial2 as _scipy_factorial2
from scipy.special import hyp1f1 as _scipy_hyp1f1


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

