r"""Restricted Kohn-Sham (RKS) DFT for closed-shell systems.

This module implements the closed-shell Restricted Kohn-Sham method,
in which every spatial orbital is doubly occupied by one alpha and one
beta electron.

Because :math:`\mathbf{P}^{\alpha} = \mathbf{P}^{\beta}`, the
per-spin Kohn-Sham matrices built by the base class are automatically
identical.

All SCF hooks are inherited from
:class:`~compute.solvers.electronic_density.kohn_sham.kohn_sham.KohnSham`;
only matrix storage is specialised.

Classes
-------
RestrictedKohnSham
    Concrete closed-shell RKS implementation.
"""

from typing import List, Optional

import numpy as np

from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.solvers.electronic_density.functionals.exchange_correlation_functional import (
    ExchangeCorrelationFunctional,
)
from q_block.compute.solvers.electronic_density.kohn_sham.kohn_sham import (
    KohnSham,
)
from q_block.compute.solvers.spin_pair import SpinPair


class RestrictedKohnSham(KohnSham):
    r"""Restricted closed-shell Kohn-Sham DFT.

    All electrons are paired; the total number of electrons must be
    even.  Each spatial orbital is occupied by exactly two electrons
    (one alpha, one beta).

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param functional: Exchange-correlation functional.
    :type functional: ExchangeCorrelationFunctional
    :param n_electrons: Total electron count (must be even).
    :type n_electrons: int
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
    :param precomputed_eri: Pre-built ERI tensor to reuse instead of
        recomputing. ``None`` triggers computation from scratch.
    :type precomputed_eri: Optional[np.ndarray]

    Attributes
    ----------
    n_occ : int
        Number of doubly-occupied spatial orbitals
        (:math:`N_{occ} = N_{elec}/2`).
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        functional: ExchangeCorrelationFunctional,
        n_electrons: int,
        n_radial: int = 50,
        n_angular: int = 14,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
        precomputed_eri: Optional[np.ndarray] = None,
    ) -> None:
        if n_electrons % 2 != 0:
            raise ValueError(
                f"RKS requires an even number of electrons; "
                f"got {n_electrons}."
            )
        n_occ = n_electrons // 2
        super().__init__(
            cgtos,
            functional=functional,
            n_alpha=n_occ,
            n_beta=n_occ,
            n_radial=n_radial,
            n_angular=n_angular,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
            precomputed_eri=precomputed_eri,
        )
        self.n_occ: int = n_occ

    @property
    def _shared_spin(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Matrix storage
    # ------------------------------------------------------------------

    def _store_matrices(
        self,
        fock: SpinPair,
        density: SpinPair,
        C: SpinPair,
        epsilon: SpinPair,
    ) -> None:
        """Store RKS matrices in :attr:`matrices`.

        Collapses the spin pair into single matrices (they are
        identical for closed-shell) and stores the total density.

        Keys: ``S``, ``H``, ``F``, ``P``, ``C``, ``epsilon``.

        :param fock: KS matrices (shared for RKS).
        :type fock: SpinPair
        :param density: Density matrices (shared for RKS).
        :type density: SpinPair
        :param C: MO coefficients (shared for RKS).
        :type C: SpinPair
        :param epsilon: Orbital energies (shared for RKS).
        :type epsilon: SpinPair
        """
        self.matrices = {
            "S": self.S,
            "H": self.H,
            "F": fock.alpha,
            "P": density.total,
            "C": C.alpha,
            "epsilon": epsilon.alpha,
        }
