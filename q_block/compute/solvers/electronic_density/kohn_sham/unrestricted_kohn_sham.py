r"""Unrestricted Kohn-Sham (UKS) DFT for open- and closed-shell systems.

This module implements the Unrestricted Kohn-Sham method, which uses
independent spatial orbitals for alpha and beta electrons, allowing
treatment of arbitrary spin multiplicities.

All SCF hooks are inherited from
:class:`~compute.solvers.electronic_density.kohn_sham.kohn_sham.KohnSham`;
only matrix storage is specialised to expose per-spin quantities.

Classes
-------
UnrestrictedKohnSham
    Concrete UKS implementation.
"""

from typing import List, Tuple

from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.solvers.electronic_density.functionals.exchange_correlation_functional import (
    ExchangeCorrelationFunctional,
)
from q_block.compute.solvers.electronic_density.kohn_sham.kohn_sham import (
    KohnSham,
)
from q_block.compute.solvers.spin_pair import SpinPair


class UnrestrictedKohnSham(KohnSham):
    r"""Unrestricted Kohn-Sham DFT.

    Alpha and beta electrons occupy separate sets of spatial orbitals,
    enabling treatment of any spin multiplicity.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: ``(Z, (x, y, z))`` per nucleus (Bohr).
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
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
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
        if n_alpha < 0 or n_beta < 0:
            raise ValueError(
                f"Electron counts must be non-negative; "
                f"got n_alpha={n_alpha}, n_beta={n_beta}."
            )
        super().__init__(
            cgtos,
            functional=functional,
            n_alpha=n_alpha,
            n_beta=n_beta,
            n_radial=n_radial,
            n_angular=n_angular,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
        )

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
        """Store UKS per-spin matrices in :attr:`matrices`.

        Keys: ``S``, ``H``, ``F_alpha``, ``F_beta``, ``P_alpha``,
        ``P_beta``, ``C_alpha``, ``C_beta``, ``epsilon_alpha``,
        ``epsilon_beta``.

        :param fock: Per-spin KS matrices.
        :type fock: SpinPair
        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param C: Per-spin MO coefficients.
        :type C: SpinPair
        :param epsilon: Per-spin orbital energies.
        :type epsilon: SpinPair
        """
        self.matrices = {
            "S": self.S,
            "H": self.H,
            "F_alpha": fock.alpha,
            "F_beta": fock.beta,
            "P_alpha": density.alpha,
            "P_beta": density.beta,
            "C_alpha": C.alpha,
            "C_beta": C.beta,
            "epsilon_alpha": epsilon.alpha,
            "epsilon_beta": epsilon.beta,
        }
