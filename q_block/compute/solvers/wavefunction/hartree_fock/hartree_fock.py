r"""Abstract Hartree-Fock base class for all HF variants.

This module provides the :class:`HartreeFock` class, which sits between
the generic :class:`~compute.solvers.scf.SCF` template and the concrete
HF implementations (RHF, UHF, ROHF).  It factors out the pieces that
are common to every Hartree-Fock method but **not** part of the generic
SCF loop:

* Construction of Coulomb and exchange matrices from the two-electron
  repulsion integrals via efficient NumPy ``einsum`` contractions.
* A **unified spin-resolved interface**: all methods represent Fock
  matrices and density matrices as ``(alpha, beta)`` tuples.  For RHF
  the two elements are identical; for UHF they differ; ROHF overrides
  only the Fock construction (Roothaan effective Fock) and the energy.
* Default implementations of *all five* abstract SCF hooks
  (:meth:`_build_fock`, :meth:`_compute_electronic_energy`,
  :meth:`_build_density`, :meth:`_initial_density`,
  :meth:`_collect_results`) so that concrete subclasses only need to
  implement :meth:`_store_matrices`.

Classes
-------
HartreeFock
    Abstract intermediate base between :class:`SCF` and the concrete
    HF methods.
"""

import logging
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.solvers.scf import SCF
from q_block.compute.solvers.spin_pair import SpinPair

logger = logging.getLogger(__name__)


class HartreeFock(SCF):
    r"""Abstract Hartree-Fock base class.

    Extends the :class:`~compute.solvers.scf.SCF` template with
    Hartree-Fock--specific helpers shared by every HF variant.

    All methods use a unified ``(alpha, beta)`` tuple representation
    for Fock and density matrices.  The base class provides concrete
    implementations of the five SCF hooks that cover both the RHF
    (identical alpha/beta) and UHF (different alpha/beta) cases.
    ROHF overrides :meth:`_build_fock` and
    :meth:`_compute_electronic_energy` for the Roothaan effective
    Fock operator.

    Concrete subclasses only need to implement
    :meth:`_store_matrices` to decide *which* matrices are saved in
    :attr:`matrices`.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param n_alpha: Number of alpha electrons.
    :type n_alpha: int
    :param n_beta: Number of beta electrons.
    :type n_beta: int
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
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        n_alpha: int,
        n_beta: int,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
    ) -> None:
        super().__init__(
            cgtos,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
        )
        self._n_alpha: int = n_alpha
        self._n_beta: int = n_beta
        self._C: Optional[np.ndarray] = None

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
        r"""Whether alpha and beta channels are identical by construction.

        Subclasses override to ``True`` when the method guarantees
        :math:`P^{\alpha} = P^{\beta}` (e.g. closed-shell RHF).
        """
        return False

    # ==================================================================
    # Integral helpers
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
    # SCF hooks — concrete defaults (RHF / UHF)
    # ==================================================================

    def _build_fock(
        self,
        density: SpinPair,
    ) -> SpinPair:
        r"""Build alpha and beta Fock matrices.

        .. math::

            F^{\sigma}_{\mu\nu}
                = H_{\mu\nu}
                  + J_{\mu\nu}(\mathbf{P}^{total})
                  - K_{\mu\nu}(\mathbf{P}^{\sigma})

        When the density pair is *shared* (RHF), the exchange matrix
        is computed once and a shared :class:`SpinPair` is returned.

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :returns: Per-spin Fock matrices.
        :rtype: SpinPair
        """
        J = self._build_coulomb(density.total)
        K_alpha = self._build_exchange(density.alpha)

        if density.shared:
            return SpinPair(self.H + J - K_alpha)

        K_beta = self._build_exchange(density.beta)
        return SpinPair(self.H + J - K_alpha, self.H + J - K_beta)

    def _compute_electronic_energy(
        self,
        density: SpinPair,
        fock: Optional[Union[SpinPair, tuple]] = None,
    ) -> float:
        r"""Electronic energy from per-spin density and Fock matrices.

        .. math::

            E_{elec} = \tfrac{1}{2}\bigl[
                \operatorname{Tr}[\mathbf{P}^{\alpha}\,
                    (\mathbf{H} + \mathbf{F}^{\alpha})]
              + \operatorname{Tr}[\mathbf{P}^{\beta}\,
                    (\mathbf{H} + \mathbf{F}^{\beta})]\bigr]

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param fock: Per-spin Fock matrices.  May be ``None``
            for subclasses that compute the energy from cached
            matrices (e.g. ROHF).
        :type fock: Optional[Union[SpinPair, tuple]]
        :returns: Electronic energy (Hartree).
        :rtype: float
        """
        P_alpha, P_beta = density
        F_alpha, F_beta = fock
        return 0.5 * (
            np.sum(P_alpha * (self.H + F_alpha))
            + np.sum(P_beta * (self.H + F_beta))
        )

    def _build_density(
        self,
        C: tuple,
    ) -> SpinPair:
        r"""Build per-spin density matrices from MO coefficients.

        .. math::

            P^{\sigma}_{\mu\nu}
                = \sum_{i=1}^{N_{\sigma}}
                  C^{\sigma}_{\mu i}\,C^{\sigma}_{\nu i}

        When :attr:`_shared_spin` is ``True`` the beta computation
        is skipped and a shared :class:`SpinPair` is returned.

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

        The single set of MO coefficients from diagonalising
        :math:`\mathbf{H}` is used for both spin channels.

        When :attr:`_shared_spin` is ``True`` the beta computation
        is skipped and a shared :class:`SpinPair` is returned.

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
    # _collect_results  (final implementation for all HF methods)
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

        Sets :attr:`converged`, :attr:`n_iterations`,
        :attr:`e_electronic`, and :attr:`e_total`, then wraps raw
        tuples into :class:`SpinPair` and calls the subclass
        :meth:`_store_matrices` hook.

        :param converged: Whether the SCF loop converged.
        :type converged: bool
        :param n_iterations: Iterations performed.
        :type n_iterations: int
        :param e_electronic: Final electronic energy (Hartree).
        :type e_electronic: float
        :param fock: Final Fock matrices (may be a plain tuple
            after DIIS extrapolation).
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

        Each HF variant stores its own set of named matrices.
        All parameters are guaranteed to be :class:`SpinPair`
        instances.

        :param fock: Final Fock matrices.
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
            f"{status})"
        )
