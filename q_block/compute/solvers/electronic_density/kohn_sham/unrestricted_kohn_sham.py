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

import numpy as np

from q_block.compute.solvers.electronic_density.kohn_sham.kohn_sham import (
    KohnSham,
)
from q_block.compute.solvers.spin_pair import SpinPair


class UnrestrictedKohnSham(KohnSham):
    r"""Unrestricted Kohn-Sham DFT.

    Alpha and beta electrons occupy separate sets of spatial orbitals,
    enabling treatment of any spin multiplicity.

    :param \*\*kwargs: All parameters forwarded to
        :class:`~compute.solvers.electronic_density.kohn_sham.kohn_sham.KohnSham`
        (e.g. ``functional``, ``n_alpha``, ``n_beta``, ``n_radial``,
        ``n_angular``, ``cgtos``, ``convergence_threshold``,
        ``max_iterations``).
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
