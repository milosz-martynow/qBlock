r"""Kohn-Sham DFT solver base class.

This module provides the :class:`KohnSham` class, which sits between
the generic :class:`~compute.solvers.scf.SCF` template and the concrete
Kohn-Sham implementations (RKS, UKS).  It factors out the pieces that
are common to every Kohn-Sham DFT method but **not** part of the
generic SCF loop:

* Evaluation of basis functions and their gradients on the numerical
  integration grid.
* Construction of the exchange-correlation potential matrix via
  numerical integration.
* A **unified spin-resolved interface** using
  :class:`~compute.solvers.spin_pair.SpinPair`.

The Kohn-Sham equations are structurally identical to Hartree-Fock,
with the Fock matrix replaced by the Kohn-Sham effective potential:

.. math::

    F^{KS}_{\mu\nu}
        = H_{\mu\nu}
          + J_{\mu\nu}(\mathbf{P}^{total})
          + V^{xc}_{\mu\nu}
          + a_0\,K_{\mu\nu}  \quad(\text{hybrid only})

Classes
-------
KohnSham
    Abstract intermediate base between :class:`SCF` and the concrete
    KS methods.
"""

import logging
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from compute.models.basis_functions import ContractedGaussianTypeOrbital
from compute.models.integrals.numerical_grid import NumericalGrid
from compute.solvers.electronic_density.functionals.exchange_correlation_functional import (
    ExchangeCorrelationFunctional,
)
from compute.solvers.scf import SCF
from compute.solvers.spin_pair import SpinPair
from compute.utilities.mathematics import (
    get_cartesian_components,
    normalization_constant,
)

logger = logging.getLogger(__name__)


class KohnSham(SCF):
    r"""Abstract Kohn-Sham DFT base class.

    Extends the :class:`~compute.solvers.scf.SCF` template with
    Kohn-Sham--specific functionality shared by all KS variants:

    * Numerical integration grid construction.
    * Basis function evaluation on the grid.
    * Exchange-correlation potential matrix construction via numerical
      quadrature.
    * Coulomb matrix construction from the ERI tensor.
    * Optional exact (HF) exchange for hybrid functionals.

    Concrete subclasses only need to implement
    :meth:`_store_matrices` to decide *which* matrices are saved.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: ``(Z, (x, y, z))`` for each nucleus (Bohr).
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param e_nuclear: Nuclear repulsion energy (Hartree).
    :type e_nuclear: float
    :param functional: Exchange-correlation functional.
    :type functional: ExchangeCorrelationFunctional
    :param n_alpha: Number of alpha electrons.
    :type n_alpha: int
    :param n_beta: Number of beta electrons.
    :type n_beta: int
    :param n_radial: Number of radial grid points per atom.
    :type n_radial: int
    :param n_angular: Number of angular (Lebedev) points per atom.
    :type n_angular: int
    :param max_iterations: Maximum number of SCF cycles.
    :type max_iterations: int
    :param convergence_threshold: Threshold for energy change and
        DIIS error.
    :type convergence_threshold: float
    :param diis_start: SCF iteration at which to begin DIIS
        extrapolation (0-indexed).
    :type diis_start: int
    :param diis_max_vectors: Maximum number of Fock/error pairs
        stored in the DIIS subspace.
    :type diis_max_vectors: int
    :param calculation_error_metric: Error reduction method
        (``"rms"`` or ``"max_abs"``).
    :type calculation_error_metric: str

    Attributes
    ----------
    functional : ExchangeCorrelationFunctional
        The XC functional used in the calculation.
    grid : NumericalGrid
        Molecular numerical integration grid.
    phi_grid : np.ndarray
        Basis function values on the grid, shape
        ``(n_basis, n_points)``.
    dphi_grid : Optional[np.ndarray]
        Basis function gradient values on the grid, shape
        ``(3, n_basis, n_points)``.  ``None`` for LDA functionals.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        functional: ExchangeCorrelationFunctional,
        n_alpha: int,
        n_beta: int,
        n_radial: int = 50,
        n_angular: int = 14,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
    ) -> None:
        super().__init__(
            cgtos,
            nuclei,
            e_nuclear,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
        )
        self._n_alpha: int = n_alpha
        self._n_beta: int = n_beta
        self._C: Optional[np.ndarray] = None
        self.functional: ExchangeCorrelationFunctional = functional

        # ── Numerical integration grid ───────────────────────────
        logger.info(
            f"Building numerical grid ({n_radial} radial × "
            f"{n_angular} angular)..."
        )
        self.grid: NumericalGrid = NumericalGrid(
            nuclei=nuclei,
            n_radial=n_radial,
            n_angular=n_angular,
        )

        # ── Evaluate basis functions on grid ─────────────────────
        logger.info("Evaluating basis functions on grid...")
        self.phi_grid: np.ndarray = self._evaluate_basis_on_grid()
        self.dphi_grid: Optional[np.ndarray] = None
        if self.functional.needs_gradient:
            logger.info(
                "Evaluating basis function gradients on grid..."
            )
            self.dphi_grid = self._evaluate_basis_gradients_on_grid()
        logger.info(
            f"Grid evaluation complete: "
            f"{self.n_basis} basis × {self.grid.n_points} points."
        )

    # ==================================================================
    # Electron-count properties
    # ==================================================================

    @property
    def n_alpha(self) -> int:
        """Number of alpha (spin-up) electrons."""
        return self._n_alpha

    @property
    def n_beta(self) -> int:
        """Number of beta (spin-down) electrons."""
        return self._n_beta

    @property
    def n_electrons(self) -> int:
        """Total number of electrons."""
        return self._n_alpha + self._n_beta

    @property
    def _shared_spin(self) -> bool:
        r"""Whether alpha and beta channels are identical.

        Subclasses override to ``True`` for closed-shell RKS.
        """
        return False

    # ==================================================================
    # Basis function evaluation on grid
    # ==================================================================

    def _evaluate_basis_on_grid(self) -> np.ndarray:
        r"""Evaluate all basis functions at every grid point.

        Builds the matrix :math:`\phi_{\mu}(\mathbf{r}_g)` of shape
        ``(n_basis, n_points)`` where each row is one basis function
        evaluated at all grid points.

        :returns: Basis function values, shape
            ``(n_basis, n_points)``.
        :rtype: np.ndarray
        """
        n_pts = self.grid.n_points
        phi = np.zeros((self.n_basis, n_pts))

        mu = 0
        for cgto in self.cgtos:
            components = get_cartesian_components(cgto.l)
            for lx, ly, lz in components:
                for g in range(n_pts):
                    rx = self.grid.coords[g, 0]
                    ry = self.grid.coords[g, 1]
                    rz = self.grid.coords[g, 2]
                    dx = rx - cgto.center.x
                    dy = ry - cgto.center.y
                    dz = rz - cgto.center.z
                    r_sq = dx * dx + dy * dy + dz * dz
                    angular = dx ** lx * dy ** ly * dz ** lz

                    val = 0.0
                    for p in range(cgto.n_primitives):
                        alpha = cgto.exponents[p]
                        coeff = cgto.contractions[p]
                        norm = normalization_constant(
                            alpha, lx, ly, lz
                        )
                        val += (
                            coeff
                            * norm
                            * angular
                            * np.exp(-alpha * r_sq)
                        )
                    phi[mu, g] = val
                mu += 1
        return phi

    def _evaluate_basis_gradients_on_grid(self) -> np.ndarray:
        r"""Evaluate gradients of all basis functions on the grid.

        Returns a tensor of shape ``(3, n_basis, n_points)`` containing
        :math:`\partial\phi_\mu / \partial x`, :math:`\partial\phi_\mu
        / \partial y`, :math:`\partial\phi_\mu / \partial z`.

        Uses the analytical derivative of Cartesian Gaussians:

        .. math::

            \frac{\partial\phi}{\partial x}
                = l_x\,(x - R_x)^{l_x - 1}\,\ldots
                  - 2\alpha\,(x - R_x)^{l_x + 1}\,\ldots

        :returns: Basis function gradients, shape
            ``(3, n_basis, n_points)``.
        :rtype: np.ndarray
        """
        n_pts = self.grid.n_points
        dphi = np.zeros((3, self.n_basis, n_pts))
        coords = self.grid.coords  # (n_pts, 3)

        mu = 0
        for cgto in self.cgtos:
            # Displacement vectors for all grid points: (n_pts,)
            dx = coords[:, 0] - cgto.center.x
            dy = coords[:, 1] - cgto.center.y
            dz = coords[:, 2] - cgto.center.z
            r_sq = dx * dx + dy * dy + dz * dz  # (n_pts,)

            alphas = np.array(cgto.exponents)     # (n_prim,)
            coeffs = np.array(cgto.contractions)  # (n_prim,)

            components = get_cartesian_components(cgto.l)
            for lx, ly, lz in components:
                norms = np.array([
                    normalization_constant(a, lx, ly, lz)
                    for a in alphas
                ])
                # base_p[p, g] = coeff_p * norm_p * exp(-alpha_p * r_sq_g)
                base_p = (coeffs * norms)[:, np.newaxis] * np.exp(
                    -alphas[:, np.newaxis] * r_sq[np.newaxis, :]
                )  # (n_prim, n_pts)

                # Aggregate over primitives
                base_sum = base_p.sum(axis=0)  # (n_pts,)
                base_alpha_sum = (
                    alphas[:, np.newaxis] * base_p
                ).sum(axis=0)  # (n_pts,)

                ang_yz = dy ** ly * dz ** lz  # (n_pts,)
                ang_xz = dx ** lx * dz ** lz
                ang_xy = dx ** lx * dy ** ly

                # d/dx = lx * dx^(lx-1) * ang_yz * base_sum
                #      - 2 * alpha * dx^(lx+1) * ang_yz * base_alpha_sum
                gx = -2.0 * dx ** (lx + 1) * ang_yz * base_alpha_sum
                if lx > 0:
                    gx += lx * dx ** (lx - 1) * ang_yz * base_sum

                # d/dy
                gy = -2.0 * dy ** (ly + 1) * ang_xz * base_alpha_sum
                if ly > 0:
                    gy += ly * dy ** (ly - 1) * ang_xz * base_sum

                # d/dz
                gz = -2.0 * dz ** (lz + 1) * ang_xy * base_alpha_sum
                if lz > 0:
                    gz += lz * dz ** (lz - 1) * ang_xy * base_sum

                dphi[0, mu] = gx
                dphi[1, mu] = gy
                dphi[2, mu] = gz
                mu += 1
        return dphi

    # ==================================================================
    # Density and gradient on grid
    # ==================================================================

    def _density_on_grid(
        self,
        P: np.ndarray,
    ) -> np.ndarray:
        r"""Compute electron density on the grid from a density matrix.

        .. math::

            \rho(\mathbf{r}_g)
                = \sum_{\mu\nu} P_{\mu\nu}\,
                  \phi_\mu(\mathbf{r}_g)\,\phi_\nu(\mathbf{r}_g)

        :param P: Density matrix, shape ``(n_basis, n_basis)``.
        :type P: np.ndarray
        :returns: Density values, shape ``(n_points,)``.
        :rtype: np.ndarray
        """
        # rho_g = sum_mu sum_nu P_mn * phi_m(g) * phi_n(g)
        tmp = P @ self.phi_grid  # (n_basis, n_points)
        rho = np.einsum("mg,mg->g", self.phi_grid, tmp)
        return np.maximum(rho, 0.0)

    def _gradient_on_grid(
        self,
        P: np.ndarray,
    ) -> np.ndarray:
        r"""Compute density gradient on the grid.

        .. math::

            \nabla\rho(\mathbf{r}_g)
                = 2\sum_{\mu\nu} P_{\mu\nu}\,
                  \nabla\phi_\mu(\mathbf{r}_g)\,
                  \phi_\nu(\mathbf{r}_g)

        :param P: Density matrix, shape ``(n_basis, n_basis)``.
        :type P: np.ndarray
        :returns: Gradient vectors, shape ``(3, n_points)``.
        :rtype: np.ndarray
        """
        tmp = P @ self.phi_grid  # (n_basis, n_points)
        grad = 2.0 * np.einsum(
            "dmg,mg->dg", self.dphi_grid, tmp
        )
        return grad

    # ==================================================================
    # XC potential matrix
    # ==================================================================

    def _build_vxc_matrix(
        self,
        rho_alpha: np.ndarray,
        rho_beta: np.ndarray,
        vxc_alpha: np.ndarray,
        vxc_beta: np.ndarray,
    ) -> SpinPair:
        r"""Build XC potential matrices via numerical integration.

        .. math::

            V^{xc,\sigma}_{\mu\nu}
                = \sum_g w_g\,v_{xc}^\sigma(\mathbf{r}_g)\,
                  \phi_\mu(\mathbf{r}_g)\,\phi_\nu(\mathbf{r}_g)

        :param rho_alpha: Alpha density on grid.
        :type rho_alpha: np.ndarray
        :param rho_beta: Beta density on grid.
        :type rho_beta: np.ndarray
        :param vxc_alpha: Alpha XC potential on grid.
        :type vxc_alpha: np.ndarray
        :param vxc_beta: Beta XC potential on grid.
        :type vxc_beta: np.ndarray
        :returns: XC potential matrices.
        :rtype: SpinPair
        """
        w = self.grid.weights

        # V^xc_alpha_mn = sum_g w_g * v_a(g) * phi_m(g) * phi_n(g)
        weighted_a = self.phi_grid * (w * vxc_alpha)[np.newaxis, :]
        Vxc_alpha = weighted_a @ self.phi_grid.T

        if self._shared_spin:
            return SpinPair(Vxc_alpha)

        weighted_b = self.phi_grid * (w * vxc_beta)[np.newaxis, :]
        Vxc_beta = weighted_b @ self.phi_grid.T

        return SpinPair(Vxc_alpha, Vxc_beta)

    # ==================================================================
    # XC energy
    # ==================================================================

    def _compute_xc_energy(
        self,
        exc: np.ndarray,
        rho: np.ndarray,
    ) -> float:
        r"""Compute total exchange-correlation energy via quadrature.

        .. math::

            E_{xc} = \sum_g w_g\,\rho(\mathbf{r}_g)\,
                     \varepsilon_{xc}(\mathbf{r}_g)

        :param exc: XC energy density per electron, shape
            ``(n_points,)``.
        :type exc: np.ndarray
        :param rho: Total density on grid, shape ``(n_points,)``.
        :type rho: np.ndarray
        :returns: Exchange-correlation energy (Hartree).
        :rtype: float
        """
        return float(np.sum(self.grid.weights * rho * exc))

    # ==================================================================
    # Coulomb and exchange helpers
    # ==================================================================

    def _build_coulomb(self, P: np.ndarray) -> np.ndarray:
        r"""Coulomb matrix from a density matrix.

        .. math::

            J_{\mu\nu} = \sum_{\lambda\sigma}
                P_{\lambda\sigma}\,(\mu\nu|\lambda\sigma)

        :param P: Density matrix, shape ``(n_basis, n_basis)``.
        :type P: np.ndarray
        :returns: Coulomb matrix **J**.
        :rtype: np.ndarray
        """
        return np.einsum("ls,mnls->mn", P, self.eri)

    def _build_exchange(self, P: np.ndarray) -> np.ndarray:
        r"""Exchange matrix from a density matrix.

        .. math::

            K_{\mu\nu} = \sum_{\lambda\sigma}
                P_{\lambda\sigma}\,(\mu\lambda|\nu\sigma)

        :param P: Density matrix, shape ``(n_basis, n_basis)``.
        :type P: np.ndarray
        :returns: Exchange matrix **K**.
        :rtype: np.ndarray
        """
        return np.einsum("ls,mlns->mn", P, self.eri)

    # ==================================================================
    # SCF hooks — concrete defaults (RKS / UKS)
    # ==================================================================

    def _build_fock(
        self,
        density: SpinPair,
    ) -> SpinPair:
        r"""Build Kohn-Sham effective potential matrices.

        .. math::

            F^{KS,\sigma}_{\mu\nu}
                = H_{\mu\nu}
                  + J_{\mu\nu}(\mathbf{P}^{total})
                  + V^{xc,\sigma}_{\mu\nu}
                  - a_0\,K_{\mu\nu}(\mathbf{P}^{\sigma})

        The exchange term is only included for hybrid functionals
        (:math:`a_0 > 0`).

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :returns: Per-spin Kohn-Sham matrices.
        :rtype: SpinPair
        """
        # Compute density on grid
        rho_alpha_grid = self._density_on_grid(density.alpha)
        rho_beta_grid = self._density_on_grid(density.beta)

        # Compute gradient on grid (if needed)
        gamma_aa = None
        gamma_ab = None
        gamma_bb = None
        if self.functional.needs_gradient and self.dphi_grid is not None:
            grad_a = self._gradient_on_grid(density.alpha)
            grad_b = self._gradient_on_grid(density.beta)
            gamma_aa = np.sum(grad_a * grad_a, axis=0)
            gamma_ab = np.sum(grad_a * grad_b, axis=0)
            gamma_bb = np.sum(grad_b * grad_b, axis=0)

        # Evaluate XC functional
        _, vxc_a, vxc_b = self.functional.compute_exc_vxc(
            rho_alpha_grid,
            rho_beta_grid,
            gamma_aa=gamma_aa,
            gamma_ab=gamma_ab,
            gamma_bb=gamma_bb,
        )

        # Build XC potential matrix
        Vxc = self._build_vxc_matrix(
            rho_alpha_grid, rho_beta_grid, vxc_a, vxc_b
        )

        # Coulomb
        J = self._build_coulomb(density.total)

        # Hybrid exchange
        a0 = self.functional.exact_exchange_fraction
        if a0 > 0.0:
            K_alpha = self._build_exchange(density.alpha)
            F_alpha = self.H + J + Vxc.alpha - a0 * K_alpha

            if density.shared:
                return SpinPair(F_alpha)

            K_beta = self._build_exchange(density.beta)
            F_beta = self.H + J + Vxc.beta - a0 * K_beta
            return SpinPair(F_alpha, F_beta)

        # Pure DFT (no exact exchange)
        F_alpha = self.H + J + Vxc.alpha
        if density.shared:
            return SpinPair(F_alpha)

        F_beta = self.H + J + Vxc.beta
        return SpinPair(F_alpha, F_beta)

    def _compute_electronic_energy(
        self,
        density: SpinPair,
        fock: Optional[Union[SpinPair, tuple]] = None,
    ) -> float:
        r"""Kohn-Sham electronic energy.

        .. math::

            E_{elec} = \sum_\sigma \operatorname{Tr}[\mathbf{P}^\sigma
                       \mathbf{H}]
                + \tfrac{1}{2}\operatorname{Tr}[\mathbf{P}^{total}
                  \mathbf{J}]
                + E_{xc}[\rho]
                - a_0 \sum_\sigma \tfrac{1}{2}
                  \operatorname{Tr}[\mathbf{P}^\sigma
                  \mathbf{K}^\sigma]

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param fock: Unused (energy computed directly).
        :type fock: Optional[Union[SpinPair, tuple]]
        :returns: Electronic energy (Hartree).
        :rtype: float
        """
        # One-electron energy
        e_one = np.sum(density.total * self.H)

        # Coulomb energy
        J = self._build_coulomb(density.total)
        e_coulomb = 0.5 * np.sum(density.total * J)

        # XC energy on grid
        rho_a_grid = self._density_on_grid(density.alpha)
        rho_b_grid = self._density_on_grid(density.beta)
        rho_grid = rho_a_grid + rho_b_grid

        gamma_aa = None
        gamma_ab = None
        gamma_bb = None
        if self.functional.needs_gradient and self.dphi_grid is not None:
            grad_a = self._gradient_on_grid(density.alpha)
            grad_b = self._gradient_on_grid(density.beta)
            gamma_aa = np.sum(grad_a * grad_a, axis=0)
            gamma_ab = np.sum(grad_a * grad_b, axis=0)
            gamma_bb = np.sum(grad_b * grad_b, axis=0)

        exc, _, _ = self.functional.compute_exc_vxc(
            rho_a_grid,
            rho_b_grid,
            gamma_aa=gamma_aa,
            gamma_ab=gamma_ab,
            gamma_bb=gamma_bb,
        )
        e_xc = self._compute_xc_energy(exc, rho_grid)

        # Exact exchange energy (hybrid only)
        e_exact_exchange = 0.0
        a0 = self.functional.exact_exchange_fraction
        if a0 > 0.0:
            K_alpha = self._build_exchange(density.alpha)
            e_exact_exchange -= a0 * 0.5 * np.sum(
                density.alpha * K_alpha
            )
            if not density.shared:
                K_beta = self._build_exchange(density.beta)
                e_exact_exchange -= a0 * 0.5 * np.sum(
                    density.beta * K_beta
                )
            else:
                e_exact_exchange *= 2.0

        return e_one + e_coulomb + e_xc + e_exact_exchange

    def _build_density(
        self,
        C: tuple,
    ) -> SpinPair:
        r"""Build per-spin density matrices from MO coefficients.

        .. math::

            P^{\sigma}_{\mu\nu}
                = \sum_{i=1}^{N_{\sigma}}
                  C^{\sigma}_{\mu i}\,C^{\sigma}_{\nu i}

        :param C: ``(C_alpha, C_beta)`` MO coefficient matrices.
        :type C: tuple
        :returns: Per-spin density matrices.
        :rtype: SpinPair
        """
        C_alpha, C_beta = C
        self._C = C_alpha
        Ca_occ = C_alpha[:, : self._n_alpha]
        P_alpha = Ca_occ @ Ca_occ.T

        if self._shared_spin:
            return SpinPair(P_alpha)

        Cb_occ = C_beta[:, : self._n_beta]
        return SpinPair(P_alpha, Cb_occ @ Cb_occ.T)

    def _initial_density(
        self,
        C: np.ndarray,
    ) -> SpinPair:
        r"""Initial density guess from core-Hamiltonian MOs.

        :param C: MO coefficient matrix from the initial
            diagonalisation.
        :type C: np.ndarray
        :returns: Per-spin initial density matrices.
        :rtype: SpinPair
        """
        self._C = C
        Ca_occ = C[:, : self._n_alpha]
        P_alpha = Ca_occ @ Ca_occ.T

        if self._shared_spin:
            return SpinPair(P_alpha)

        Cb_occ = C[:, : self._n_beta]
        return SpinPair(P_alpha, Cb_occ @ Cb_occ.T)

    # ==================================================================
    # _collect_results
    # ==================================================================

    def _collect_results(
        self,
        converged: bool,
        n_iterations: int,
        e_electronic: float,
        fock: Union[SpinPair, tuple],
        density: SpinPair,
        C: tuple,
        epsilon: tuple,
    ) -> None:
        """Populate scalar result attributes and delegate matrix storage.

        :param converged: Whether the SCF loop converged.
        :type converged: bool
        :param n_iterations: Iterations performed.
        :type n_iterations: int
        :param e_electronic: Final electronic energy (Hartree).
        :type e_electronic: float
        :param fock: Final KS matrices.
        :type fock: Union[SpinPair, tuple]
        :param density: Final per-spin density matrices.
        :type density: SpinPair
        :param C: Final MO coefficients.
        :type C: tuple
        :param epsilon: Final orbital energies.
        :type epsilon: tuple
        """
        self.converged = converged
        self.n_iterations = n_iterations
        self.e_electronic = e_electronic
        self.e_total = e_electronic + self.e_nuclear
        logger.info(
            f"{type(self).__name__} results: "
            f"converged={converged}, iterations={n_iterations}, "
            f"E_electronic={e_electronic:.10f}, "
            f"E_nuclear={self.e_nuclear:.10f}, "
            f"E_total={self.e_total:.10f}"
        )
        self._store_matrices(
            SpinPair.wrap(fock),
            SpinPair.wrap(density),
            SpinPair.wrap(C),
            SpinPair.wrap(epsilon),
        )

    @abstractmethod
    def _store_matrices(
        self,
        fock: SpinPair,
        density: SpinPair,
        C: SpinPair,
        epsilon: SpinPair,
    ) -> None:
        """Populate :attr:`matrices` (and optionally :attr:`extra`).

        :param fock: Final KS matrices.
        :type fock: SpinPair
        :param density: Final per-spin density matrices.
        :type density: SpinPair
        :param C: Final MO coefficients.
        :type C: SpinPair
        :param epsilon: Final orbital energies.
        :type epsilon: SpinPair
        """

    # ==================================================================
    # repr
    # ==================================================================

    def __repr__(self) -> str:
        if self.converged is None:
            status = "not yet run"
        elif self.converged:
            status = f"converged, e_total={self.e_total:.10f}"
        else:
            status = "not converged"
        return (
            f"{type(self).__name__}("
            f"n_basis={self.n_basis}, "
            f"n_alpha={self._n_alpha}, n_beta={self._n_beta}, "
            f"functional={self.functional.name}, "
            f"{status})"
        )
