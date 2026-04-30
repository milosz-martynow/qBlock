r"""
TODO: Answer whether this should not live somewhere in coordinates.py

Numerical integration grid for Density Functional Theory.

This module provides the :class:`NumericalGrid` class that generates
Becke-partitioned molecular grids for numerical integration of the
exchange-correlation energy and potential in DFT calculations.

Each atom receives a product grid:

* **Radial** — Euler-Maclaurin or Gauss-Chebyshev quadrature, with
  Bragg-Slater atomic radii for scaling.
* **Angular** — Lebedev quadrature providing exact integration of
  spherical harmonics up to a specified order.

Multi-centre integration uses the Becke partitioning scheme with
Becke's atomic-size adjustments:

.. math::

    w_A(\mathbf{r})
        = \frac{P_A(\mathbf{r})}{\sum_B P_B(\mathbf{r})}

where :math:`P_A` is the product of step functions constructed from
confocal elliptic coordinates between atom pairs.

Classes
-------
NumericalGrid
    Molecular integration grid with Becke partitioning.
"""

import logging
from typing import List, Tuple

import numpy as np

from q_block.compute.environment.constants.numerical import (
    BRAGG_SLATER_RADII,
    LEBEDEV_GRIDS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Radial quadrature
# ──────────────────────────────────────────────────────────────────────


def _euler_maclaurin_radial(
    n_radial: int,
    rm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Euler-Maclaurin radial quadrature.

    Maps uniform points on :math:`(0, 1)` to :math:`(0, \infty)` via
    the Murray-Handy-Laming transformation:

    .. math::

        r_i = r_m \frac{x_i^2}{(1 - x_i)^2},
        \qquad
        w_i = \frac{2 r_m x_i^5}{(1 - x_i)^7} \cdot \frac{1}{n+1}

    where :math:`x_i = i / (n + 1)` and :math:`r_m` is the
    Bragg-Slater radius.

    :param n_radial: Number of radial points.
    :type n_radial: int
    :param rm: Bragg-Slater radius for the atom (Bohr).
    :type rm: float
    :returns: ``(radii, weights)`` each of shape ``(n_radial,)``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    i = np.arange(1, n_radial + 1, dtype=float)
    x = i / (n_radial + 1.0)
    r = rm * x * x / (1.0 - x) ** 2
    dr = 2.0 * rm * x / (1.0 - x) ** 3 / (n_radial + 1.0)
    w = 4.0 * np.pi * r * r * dr
    return r, w


# ──────────────────────────────────────────────────────────────────────
# Becke partitioning
# ──────────────────────────────────────────────────────────────────────


def _becke_step(mu: np.ndarray) -> np.ndarray:
    r"""Becke's smoothed step function (three iterations).

    Applies the polynomial smoothing :math:`s(\mu) = \tfrac{3}{2}\mu
    - \tfrac{1}{2}\mu^3` three times.

    :param mu: Confocal elliptic coordinate values in ``[-1, 1]``.
    :type mu: np.ndarray
    :returns: Smoothed step function values.
    :rtype: np.ndarray
    """
    for _ in range(3):
        mu = 1.5 * mu - 0.5 * mu ** 3
    return mu


def _becke_partition_weights(
    grid_coords: np.ndarray,
    nuclei: List[Tuple[int, Tuple[float, float, float]]],
    atom_index: int,
) -> np.ndarray:
    r"""Compute Becke partition weights for a single atom's grid.

    :param grid_coords: Grid point coordinates, shape
        ``(n_points, 3)``.
    :type grid_coords: np.ndarray
    :param nuclei: ``(Z, (x, y, z))`` for each nucleus.
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param atom_index: Index of the atom whose partition weights
        are being computed.
    :type atom_index: int
    :returns: Partition weights, shape ``(n_points,)``.
    :rtype: np.ndarray
    """
    n_atoms = len(nuclei)
    n_pts = grid_coords.shape[0]
    centers = np.array([nuc[1] for nuc in nuclei])

    # Distances from grid points to each nucleus
    # dist[i, A] = |r_i - R_A|
    dist = np.zeros((n_pts, n_atoms))
    for a in range(n_atoms):
        diff = grid_coords - centers[a]
        dist[:, a] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Inter-nuclear distances
    r_ab = np.zeros((n_atoms, n_atoms))
    for a in range(n_atoms):
        for b in range(a + 1, n_atoms):
            d = np.sqrt(np.sum((centers[a] - centers[b]) ** 2))
            r_ab[a, b] = d
            r_ab[b, a] = d

    # Becke partition for each point
    P = np.ones((n_pts, n_atoms))
    for a in range(n_atoms):
        for b in range(n_atoms):
            if a == b:
                continue
            mu_ab = (dist[:, a] - dist[:, b]) / r_ab[a, b]
            s_ab = 0.5 * (1.0 - _becke_step(mu_ab))
            P[:, a] *= s_ab

    # Normalize
    P_sum = np.sum(P, axis=1)
    mask = P_sum > 1e-30
    weights = np.zeros(n_pts)
    weights[mask] = P[mask, atom_index] / P_sum[mask]
    return weights


# ======================================================================
# N U M E R I C A L   G R I D
# ======================================================================


class NumericalGrid:
    r"""Molecular integration grid with Becke partitioning.

    Constructs a molecular grid by combining atom-centred radial +
    angular product grids and applying Becke partition weights.

    After construction the grid points and weights are available as
    :attr:`coords` and :attr:`weights`.

    :param nuclei: ``(Z, (x, y, z))`` per nucleus (Bohr).
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param n_radial: Number of radial quadrature points per atom.
    :type n_radial: int
    :param n_angular: Number of angular (Lebedev) points per radial
        shell.  Supported: 6, 14, 26.
    :type n_angular: int

    Attributes
    ----------
    coords : np.ndarray
        Grid point Cartesian coordinates, shape ``(n_points, 3)``.
    weights : np.ndarray
        Integration weights, shape ``(n_points,)``.
    n_points : int
        Total number of grid points.
    """

    def __init__(
        self,
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        n_radial: int = 50,
        n_angular: int = 14,
    ) -> None:
        if n_angular not in LEBEDEV_GRIDS:
            raise ValueError(
                f"Unsupported n_angular={n_angular}; "
                f"supported: {sorted(LEBEDEV_GRIDS.keys())}."
            )
        if n_radial < 1:
            raise ValueError(
                f"n_radial must be >= 1; got {n_radial}."
            )

        self._nuclei = nuclei
        self._n_radial = n_radial
        self._n_angular = n_angular

        self.coords: np.ndarray
        self.weights: np.ndarray
        self.n_points: int

        self._build_grid()

    def _build_grid(self) -> None:
        """Build the full molecular grid."""
        ang_pts, ang_wts = LEBEDEV_GRIDS[self._n_angular]()

        all_coords = []
        all_weights = []

        for atom_idx, (Z, (cx, cy, cz)) in enumerate(
            self._nuclei
        ):
            rm = BRAGG_SLATER_RADII.get(Z, 1.5)
            r, r_w = _euler_maclaurin_radial(
                self._n_radial, rm
            )

            # Product grid: radial x angular
            n_local = self._n_radial * self._n_angular
            local_coords = np.zeros((n_local, 3))
            local_weights = np.zeros(n_local)

            idx = 0
            for ir in range(self._n_radial):
                for ia in range(self._n_angular):
                    local_coords[idx, 0] = (
                        cx + r[ir] * ang_pts[ia, 0]
                    )
                    local_coords[idx, 1] = (
                        cy + r[ir] * ang_pts[ia, 1]
                    )
                    local_coords[idx, 2] = (
                        cz + r[ir] * ang_pts[ia, 2]
                    )
                    local_weights[idx] = (
                        r_w[ir] * ang_wts[ia]
                    )
                    idx += 1

            # Apply Becke partitioning
            if len(self._nuclei) > 1:
                becke_w = _becke_partition_weights(
                    local_coords, self._nuclei, atom_idx
                )
                local_weights *= becke_w

            all_coords.append(local_coords)
            all_weights.append(local_weights)

        self.coords = np.vstack(all_coords)
        self.weights = np.concatenate(all_weights)
        self.n_points = len(self.weights)

        logger.info(
            f"Numerical grid: {len(self._nuclei)} atoms, "
            f"{self.n_points} points "
            f"({self._n_radial} radial × "
            f"{self._n_angular} angular)"
        )

    def __repr__(self) -> str:
        return (
            f"NumericalGrid(n_points={self.n_points}, "
            f"n_radial={self._n_radial}, "
            f"n_angular={self._n_angular})"
        )
