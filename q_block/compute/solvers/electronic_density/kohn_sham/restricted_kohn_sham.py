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

from q_block.compute.solvers.electronic_density.kohn_sham.kohn_sham import (
    KohnSham,
)
from q_block.compute.solvers.spin_pair import SpinPair


class RestrictedKohnSham(KohnSham):
    r"""Restricted closed-shell Kohn-Sham DFT.

    All electrons are paired; the total number of electrons must be
    even.  Each spatial orbital is occupied by exactly two electrons
    (one alpha, one beta).

    :param n_electrons: Total electron count (must be even).
    :type n_electrons: int
    :param \*\*kwargs: All parameters forwarded to
        :class:`~compute.solvers.electronic_density.kohn_sham.kohn_sham.KohnSham`
        (e.g. ``functional``, ``n_radial``, ``n_angular``, ``cgtos``,
        ``convergence_threshold``, ``max_iterations``).
    """

    def __init__(
        self,
        n_electrons: int,
        **kwargs,
    ) -> None:
        if n_electrons % 2 != 0:
            raise ValueError(
                f"RKS requires an even number of electrons; "
                f"got {n_electrons}."
            )
        n_occ = n_electrons // 2
        super().__init__(n_alpha=n_occ, n_beta=n_occ, **kwargs)

    @property
    def n_occ(self) -> int:
        """Number of doubly-occupied spatial orbitals."""
        return self.n_alpha

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
