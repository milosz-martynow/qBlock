r"""Unrestricted Hartree-Fock (UHF) for open- and closed-shell systems.

This module implements the Unrestricted Hartree-Fock method, which uses
independent spatial orbitals for alpha and beta electrons, allowing
treatment of arbitrary spin multiplicities.

All SCF hooks are inherited from :class:`~compute.solvers.wavefunction
.hartree_fock.hartree_fock.HartreeFock`; only matrix storage
is specialised to expose per-spin quantities.

Classes
-------
UnrestrictedHartreeFock
    Concrete UHF implementation.
"""

from typing import List, Tuple

from compute.models.basis_functions import ContractedGaussianTypeOrbital
from compute.solvers.spin_pair import SpinPair
from compute.solvers.wavefunction.hartree_fock.hartree_fock import HartreeFock


class UnrestrictedHartreeFock(HartreeFock):
    r"""Unrestricted Hartree-Fock.

    Alpha and beta electrons occupy separate sets of spatial orbitals,
    enabling treatment of any spin multiplicity.

    The base-class ``(alpha, beta)`` tuple interface is used directly —
    unlike RHF, the two elements generally differ.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: ``(Z, (x, y, z))`` per nucleus (Bohr).
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param e_nuclear: Nuclear repulsion energy (Hartree).
    :type e_nuclear: float
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
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        n_alpha: int,
        n_beta: int,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
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
            nuclei,
            e_nuclear,
            n_alpha=n_alpha,
            n_beta=n_beta,
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
        """Store UHF per-spin matrices in :attr:`matrices`.

        Keys: ``S``, ``H``, ``F_alpha``, ``F_beta``, ``P_alpha``,
        ``P_beta``, ``C_alpha``, ``C_beta``, ``epsilon_alpha``,
        ``epsilon_beta``.

        :param fock: Per-spin Fock matrices.
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
