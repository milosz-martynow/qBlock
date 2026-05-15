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

All SCF hooks are inherited from :class:`~compute.solvers.wavefunction
.hartree_fock.hartree_fock.HartreeFock`; only matrix storage
is specialised.

Classes
-------
RestrictedHartreeFock
    Concrete closed-shell RHF implementation.
"""

from q_block.compute.solvers.spin_pair import SpinPair
from q_block.compute.solvers.wavefunction.hartree_fock.hartree_fock import HartreeFock


class RestrictedHartreeFock(HartreeFock):
    r"""Restricted closed-shell Hartree-Fock.

    All electrons are paired; the total number of electrons must be
    even.  Each spatial orbital is occupied by exactly two electrons
    (one alpha, one beta).

    Internally the ``(alpha, beta)`` tuple interface of the base
    class is used unchanged — both elements are identical for a
    closed-shell system.  :meth:`_store_matrices` collapses them
    into the conventional single-matrix representation.

    :param n_electrons: Total electron count (must be even).
    :type n_electrons: int
    :param \*\*kwargs: All parameters forwarded to
        :class:`~compute.solvers.scf.SCF` via
        :class:`~compute.solvers.wavefunction.hartree_fock.hartree_fock.HartreeFock`
        (e.g. ``cgtos``, ``convergence_threshold``, ``max_iterations``).
    """

    def __init__(
        self,
        n_electrons: int,
        **kwargs,
    ) -> None:
        if n_electrons % 2 != 0:
            raise ValueError(
                f"RHF requires an even number of electrons; " f"got {n_electrons}."
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
