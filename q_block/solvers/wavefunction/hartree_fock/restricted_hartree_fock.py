r"""Restricted Hartree-Fock (RHF) for closed-shell systems.

This module implements the closed-shell Restricted Hartree-Fock method,
in which every spatial orbital is doubly occupied by one alpha and one
beta electron.

Because :math:`\mathbf{P}^{\alpha} = \mathbf{P}^{\beta}`, the
per-spin Fock matrices built by the base class are automatically
identical, reducing to the familiar closed-shell expression:

.. math::

    F_{\mu\nu} = H_{\mu\nu} + J_{\mu\nu}(\mathbf{P})
        - \tfrac{1}{2}\,K_{\mu\nu}(\mathbf{P})

All SCF hooks are inherited from :class:`~q_block.solvers.wavefunction
.hartree_fock.hartree_fock.HartreeFock`; only matrix storage
is specialised.

Classes
-------
RestrictedHartreeFock
    Concrete closed-shell RHF implementation.
"""

from typing import List, Tuple

from q_block.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.solvers.spin_pair import SpinPair
from q_block.solvers.wavefunction.hartree_fock.hartree_fock import HartreeFock


class RestrictedHartreeFock(HartreeFock):
    r"""Restricted closed-shell Hartree-Fock.

    All electrons are paired; the total number of electrons must be
    even.  Each spatial orbital is occupied by exactly two electrons
    (one alpha, one beta).

    Internally the ``(alpha, beta)`` tuple interface of the base
    class is used unchanged — both elements are identical for a
    closed-shell system.  :meth:`_store_matrices` collapses them
    into the conventional single-matrix representation.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: ``(Z, (x, y, z))`` per nucleus (Bohr).
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param e_nuclear: Nuclear repulsion energy (Hartree).
    :type e_nuclear: float
    :param n_electrons: Total electron count (must be even).
    :type n_electrons: int
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
    n_occ : int
        Number of doubly-occupied spatial orbitals
        (:math:`N_{occ} = N_{elec}/2`).
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        n_electrons: int,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
    ) -> None:
        if n_electrons % 2 != 0:
            raise ValueError(
                f"RHF requires an even number of electrons; " f"got {n_electrons}."
            )
        n_occ = n_electrons // 2
        super().__init__(
            cgtos,
            nuclei,
            e_nuclear,
            n_alpha=n_occ,
            n_beta=n_occ,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
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
        """Store RHF matrices in :attr:`matrices`.

        Collapses the spin pair into single matrices (they are
        identical for closed-shell) and stores the total density.

        Keys: ``S``, ``H``, ``F``, ``P``, ``C``, ``epsilon``.

        :param fock: Fock matrices (shared for RHF).
        :type fock: SpinPair
        :param density: Density matrices (shared for RHF).
        :type density: SpinPair
        :param C: MO coefficients (shared for RHF).
        :type C: SpinPair
        :param epsilon: Orbital energies (shared for RHF).
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
