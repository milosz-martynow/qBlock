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

from q_block.compute.solvers.spin_pair import SpinPair
from q_block.compute.solvers.wavefunction.hartree_fock.hartree_fock import HartreeFock


class UnrestrictedHartreeFock(HartreeFock):
    r"""Unrestricted Hartree-Fock.

    Alpha and beta electrons occupy separate sets of spatial orbitals,
    enabling treatment of any spin multiplicity.

    The base-class ``(alpha, beta)`` tuple interface is used directly —
    unlike RHF, the two elements generally differ.

    :param \*\*kwargs: All parameters forwarded to
        :class:`~compute.solvers.wavefunction.hartree_fock.hartree_fock.HartreeFock`
        (e.g. ``n_alpha``, ``n_beta``, ``cgtos``,
        ``convergence_threshold``, ``max_iterations``).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
