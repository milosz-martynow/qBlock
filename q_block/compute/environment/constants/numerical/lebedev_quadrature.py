"""Lebedev angular quadrature grids.

This module provides generator functions for Lebedev quadrature rules
on the unit sphere.  Each function returns ``(points, weights)`` where
``points`` has shape ``(N, 3)`` (Cartesian coordinates on the unit
sphere) and ``weights`` has shape ``(N,)`` and sums to 1.

The grids integrate spherical harmonics exactly up to a given order
:math:`l`:

===============  =========  =====================
Function         Points     Exact up to
===============  =========  =====================
``lebedev_6``    6          :math:`l \\leq 3`
``lebedev_14``   14         :math:`l \\leq 5`
``lebedev_26``   26         :math:`l \\leq 7`
===============  =========  =====================

References
----------
V. I. Lebedev, "Values of the nodes and weights of ninth to
seventeenth order Gauss-Markov quadrature formulae invariant under the
octahedron group with inversion," *Zh. Vȳchisl. Mat. Mat. Fiz.*,
**15** (1), 48-54, 1975.

V. I. Lebedev and D. N. Laikov, "A quadrature formula for the sphere
of the 131st algebraic order of accuracy," *Doklady Mathematics*,
**59** (3), 477-481, 1999.
"""

from typing import Dict, Tuple

import numpy as np


def lebedev_6() -> Tuple[np.ndarray, np.ndarray]:
    """6-point Lebedev quadrature (exact for l <= 3).

    :returns: ``(points, weights)`` where points has shape ``(6, 3)``
        and weights has shape ``(6,)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    wts = np.full(6, 1.0 / 6.0)
    return pts, wts


def lebedev_14() -> Tuple[np.ndarray, np.ndarray]:
    """14-point Lebedev quadrature (exact for l <= 5).

    :returns: ``(points, weights)`` where points has shape ``(14, 3)``
        and weights has shape ``(14,)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pts_list = []
    wts_list = []

    # 6 octahedral vertices
    w1 = 1.0 / 15.0
    for sign in [1.0, -1.0]:
        for axis in range(3):
            pt = [0.0, 0.0, 0.0]
            pt[axis] = sign
            pts_list.append(pt)
            wts_list.append(w1)

    # 8 cube vertices
    w2 = 3.0 / 40.0
    val = 1.0 / np.sqrt(3.0)
    for sx in [1.0, -1.0]:
        for sy in [1.0, -1.0]:
            for sz in [1.0, -1.0]:
                pts_list.append(
                    [sx * val, sy * val, sz * val]
                )
                wts_list.append(w2)

    return np.array(pts_list), np.array(wts_list)


def lebedev_26() -> Tuple[np.ndarray, np.ndarray]:
    """26-point Lebedev quadrature (exact for l <= 7).

    :returns: ``(points, weights)`` where points has shape ``(26, 3)``
        and weights has shape ``(26,)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    pts_list = []
    wts_list = []

    # 6 octahedral vertices
    w1 = 1.0 / 21.0
    for sign in [1.0, -1.0]:
        for axis in range(3):
            pt = [0.0, 0.0, 0.0]
            pt[axis] = sign
            pts_list.append(pt)
            wts_list.append(w1)

    # 12 edge midpoints
    w2 = 4.0 / 105.0
    val = 1.0 / np.sqrt(2.0)
    for i in range(3):
        for j in range(i + 1, 3):
            for si in [1.0, -1.0]:
                for sj in [1.0, -1.0]:
                    pt = [0.0, 0.0, 0.0]
                    pt[i] = si * val
                    pt[j] = sj * val
                    pts_list.append(pt)
                    wts_list.append(w2)

    # 8 cube vertices
    w3 = 27.0 / 840.0
    val3 = 1.0 / np.sqrt(3.0)
    for sx in [1.0, -1.0]:
        for sy in [1.0, -1.0]:
            for sz in [1.0, -1.0]:
                pts_list.append(
                    [sx * val3, sy * val3, sz * val3]
                )
                wts_list.append(w3)

    return np.array(pts_list), np.array(wts_list)


LEBEDEV_GRIDS: Dict[int, callable] = {
    6: lebedev_6,
    14: lebedev_14,
    26: lebedev_26,
}
