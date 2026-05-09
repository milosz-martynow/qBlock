r"""Restricted Open-Shell Kohn-Sham (ROKS) DFT for open-shell systems.

This module implements the Restricted Open-Shell Kohn-Sham method using
the Roothaan effective Fock approach (Guest & Saunders, *Mol. Phys.* **28**,
819, 1974), analogous to ROHF but with the Hartree-Fock exchange replaced
by the Kohn-Sham effective potential.

The orbital space is partitioned into three subspaces:

* **Closed** – doubly-occupied orbitals (indices :math:`0 \ldots N_c - 1`).
* **Open** – singly-occupied orbitals
  (indices :math:`N_c \ldots N_c + N_o - 1`), occupied by alpha only.
* **Virtual** – unoccupied orbitals.

Alpha and beta electrons share the same set of spatial MOs.  The per-spin
density matrices are:

.. math::

    P^{\alpha}_{\mu\nu} = \sum_{i=1}^{N_c + N_o} C_{\mu i}\,C_{\nu i}

    P^{\beta}_{\mu\nu} = \sum_{i=1}^{N_c} C_{\mu i}\,C_{\nu i}

The per-spin Kohn-Sham Fock matrices

.. math::

    F^{\sigma}_{KS,\mu\nu}
        = H_{\mu\nu}
          + J_{\mu\nu}(\mathbf{P}^{total})
          + V^{xc,\sigma}_{\mu\nu}
          - a_0\,K_{\mu\nu}(\mathbf{P}^{\sigma})

are built by the base
:class:`~q_block.compute.solvers.electronic_density.kohn_sham.kohn_sham.KohnSham`
class and then combined into the Roothaan effective Fock operator
block-wise in the MO basis:

.. math::

    F^{eff}_{MO} =
    \begin{pmatrix}
        \mathbf{F}^{\beta}_{cc}  &
            \mathbf{F}^{\beta}_{co}  &
            \mathbf{F}^{\beta}_{cv}  \\
        \mathbf{F}^{\beta}_{oc}  &
            \mathbf{F}^{\alpha}_{oo} &
            \mathbf{F}^{\alpha}_{ov} \\
        \mathbf{F}^{\beta}_{vc}  &
            \mathbf{F}^{\alpha}_{vo} &
            \tfrac{1}{2}(\mathbf{F}^{\alpha}
                         +\mathbf{F}^{\beta})_{vv}
    \end{pmatrix}

and back-transformed to the AO basis via
:math:`\mathbf{F}^{eff} = (\mathbf{S}\mathbf{C})\,\mathbf{F}^{eff}_{MO}\,
(\mathbf{S}\mathbf{C})^T`.

The electronic energy uses the standard KS expression:

.. math::

    E_{elec}
        = \operatorname{Tr}[\mathbf{P}^{total}\mathbf{H}]
          + \tfrac{1}{2}\operatorname{Tr}[\mathbf{P}^{total}\mathbf{J}]
          + E_{xc}[\rho^{\alpha}, \rho^{\beta}]
          - a_0 \sum_{\sigma} \tfrac{1}{2}
            \operatorname{Tr}[\mathbf{P}^{\sigma}\mathbf{K}^{\sigma}]

Classes
-------
RestrictedOpenShellKohnSham
    Concrete ROKS implementation.
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


class RestrictedOpenShellKohnSham(KohnSham):
    r"""Restricted Open-Shell Kohn-Sham DFT.

    Doubly-occupied (closed) and singly-occupied (open) orbitals share
    a single set of spatial MOs, eliminating spin contamination.  The
    Roothaan effective Fock matrix is assembled from the per-spin KS
    Fock matrices and used to drive the SCF loop.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param functional: Exchange-correlation functional.
    :type functional: ExchangeCorrelationFunctional
    :param n_closed: Number of doubly-occupied spatial orbitals.
    :type n_closed: int
    :param n_open: Number of singly-occupied spatial orbitals.
    :type n_open: int
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
    n_closed : int
        Number of doubly-occupied (closed-shell) spatial orbitals.
    n_open : int
        Number of singly-occupied (open-shell) spatial orbitals.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        functional: ExchangeCorrelationFunctional,
        n_closed: int,
        n_open: int,
        n_radial: int = 50,
        n_angular: int = 14,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
        precomputed_eri: Optional[np.ndarray] = None,
    ) -> None:
        if n_closed < 0 or n_open < 0:
            raise ValueError(
                f"Orbital counts must be non-negative; "
                f"got n_closed={n_closed}, n_open={n_open}."
            )
        if n_open == 0:
            raise ValueError(
                "ROKS requires at least one open-shell orbital; "
                "use RestrictedKohnSham for closed-shell systems."
            )
        n_alpha = n_closed + n_open
        n_beta = n_closed
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
            precomputed_eri=precomputed_eri,
        )
        self.n_closed: int = n_closed
        self.n_open: int = n_open

        # ── Internal caches (set by _build_fock, read by _store_matrices) ──
        self._F_alpha: Optional[np.ndarray] = None
        self._F_beta: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # SCF hooks (overrides)
    # ------------------------------------------------------------------

    def _build_fock(
        self,
        density: SpinPair,
    ) -> SpinPair:
        r"""Build the Roothaan effective KS Fock matrix.

        1. Delegates to the base class to get the physical per-spin KS
           matrices :math:`(\mathbf{F}^{\alpha}_{KS},
           \mathbf{F}^{\beta}_{KS})`.
        2. Transforms both to the MO basis using cached coefficients.
        3. Assembles the effective Fock block-wise (Guest-Saunders).
        4. Back-transforms to the AO basis.

        Returns a shared :class:`SpinPair` containing the single
        effective Fock matrix so that the SCF loop and DIIS treat
        ROKS as a one-Fock-matrix method.

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :returns: Shared effective Fock matrix pair.
        :rtype: SpinPair
        """
        # ── Physical alpha / beta KS Fock matrices ───────────────────
        ks_fock = super()._build_fock(density)
        F_alpha = ks_fock.alpha
        F_beta = ks_fock.beta
        self._F_alpha = F_alpha
        self._F_beta = F_beta

        # ── Roothaan effective Fock in MO basis ──────────────────────
        C = self._C
        n = self.n_basis
        nc = self.n_closed
        no = self.n_open

        Fa_mo = C.T @ F_alpha @ C
        Fb_mo = C.T @ F_beta @ C

        F_eff_mo = np.zeros((n, n))

        c = slice(0, nc)
        o = slice(nc, nc + no)
        v = slice(nc + no, n)

        # Diagonal blocks
        F_eff_mo[c, c] = Fb_mo[c, c]
        F_eff_mo[o, o] = Fa_mo[o, o]
        F_eff_mo[v, v] = 0.5 * (Fa_mo[v, v] + Fb_mo[v, v])

        # Off-diagonal blocks (closed <-> open)
        F_eff_mo[c, o] = Fb_mo[c, o]
        F_eff_mo[o, c] = Fb_mo[o, c]

        # Off-diagonal blocks (closed <-> virtual)
        F_eff_mo[c, v] = Fb_mo[c, v]
        F_eff_mo[v, c] = Fb_mo[v, c]

        # Off-diagonal blocks (open <-> virtual)
        F_eff_mo[o, v] = Fa_mo[o, v]
        F_eff_mo[v, o] = Fa_mo[v, o]

        # ── Back-transform to AO basis ────────────────────────────────
        SC = self.S @ C
        F_eff = SC @ F_eff_mo @ SC.T
        return SpinPair(F_eff)

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
        """Store ROKS matrices in :attr:`matrices`.

        Keys: ``S``, ``H``, ``F_eff``, ``P``, ``F_alpha``, ``F_beta``,
        ``P_alpha``, ``P_beta``, ``C``, ``epsilon``.

        :param fock: Effective KS Fock matrices.
        :type fock: SpinPair
        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param C: MO coefficients (identical for ROKS).
        :type C: SpinPair
        :param epsilon: Orbital energies from the effective Fock.
        :type epsilon: SpinPair
        """
        self.matrices = {
            "S": self.S,
            "H": self.H,
            "F_eff": fock.alpha,
            "P": density.total,
            "F_alpha": self._F_alpha,
            "F_beta": self._F_beta,
            "P_alpha": density.alpha,
            "P_beta": density.beta,
            "C": C.alpha,
            "epsilon": epsilon.alpha,
        }
