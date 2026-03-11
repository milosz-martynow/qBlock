r"""Restricted Open-Shell Hartree-Fock (ROHF).

This module implements the Restricted Open-Shell Hartree-Fock method,
which uses a single set of spatial orbitals shared by alpha and beta
electrons while allowing singly-occupied (open-shell) orbitals.

The orbital space is partitioned into three subspaces:

* **Closed** – doubly-occupied orbitals (indices :math:`1 \ldots N_c`).
* **Open** – singly-occupied orbitals
  (indices :math:`N_c{+}1 \ldots N_c{+}N_o`), occupied by alpha only.
* **Virtual** – unoccupied orbitals.

Because alpha and beta electrons share the same spatial MOs, the
standard Fock matrix is replaced by the Roothaan effective Fock
operator (Guest & Saunders, *Mol. Phys.* **28**, 819, 1974).  In the
MO basis the effective Fock is assembled block-wise:

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

The electronic energy uses the general HF expression with the
*physical* (not effective) per-spin Fock matrices:

.. math::

    E_{elec} = \tfrac{1}{2}\bigl[
        \operatorname{Tr}[\mathbf{P}^{\alpha}\,
            (\mathbf{H} + \mathbf{F}^{\alpha})]
      + \operatorname{Tr}[\mathbf{P}^{\beta}\,
            (\mathbf{H} + \mathbf{F}^{\beta})]\bigr]

Classes
-------
RestrictedOpenShellHartreeFock
    Concrete ROHF implementation.
"""

from typing import List, Tuple

import numpy as np

from q_block.methods.wavefunction.hartree_fock.hartree_fock import (
    HartreeFock,
    SpinPair,
)
from q_block.theory.basis_functions import ContractedGaussianTypeOrbital


class RestrictedOpenShellHartreeFock(HartreeFock):
    r"""Restricted Open-Shell Hartree-Fock.

    Doubly-occupied (closed) orbitals share the same spatial part;
    singly-occupied (open) orbitals carry unpaired alpha electrons.
    A single set of spatial MOs is optimised via the Roothaan effective
    Fock matrix.

    Uses the unified ``(alpha, beta)`` tuple interface of the base
    class.  Overrides :meth:`_build_fock` to construct the Roothaan
    effective Fock (returned as ``(F_eff, F_eff)``), and overrides
    :meth:`_compute_electronic_energy` to use the physical per-spin
    Fock matrices instead of the effective one.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: ``(Z, (x, y, z))`` per nucleus (Bohr).
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param e_nuclear: Nuclear repulsion energy (Hartree).
    :type e_nuclear: float
    :param n_closed: Number of doubly-occupied spatial orbitals.
    :type n_closed: int
    :param n_open: Number of singly-occupied spatial orbitals.
    :type n_open: int
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
    n_closed : int
        Number of doubly-occupied (closed-shell) spatial orbitals.
    n_open : int
        Number of singly-occupied (open-shell) spatial orbitals.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        n_closed: int,
        n_open: int,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
    ) -> None:
        if n_closed < 0 or n_open < 0:
            raise ValueError(
                f"Orbital counts must be non-negative; "
                f"got n_closed={n_closed}, n_open={n_open}."
            )
        n_alpha = n_closed + n_open
        n_beta = n_closed
        super().__init__(
            cgtos, nuclei, e_nuclear,
            n_alpha=n_alpha, n_beta=n_beta,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            diis_start=diis_start,
            diis_max_vectors=diis_max_vectors,
            calculation_error_metric=calculation_error_metric,
        )
        self.n_closed: int = n_closed
        self.n_open: int = n_open

        # ── Internal caches (set by _build_fock, used by energy) ─────
        self._F_alpha: np.ndarray | None = None
        self._F_beta: np.ndarray | None = None

    # ------------------------------------------------------------------
    # SCF hooks (overrides)
    # ------------------------------------------------------------------

    def _build_fock(
        self,
        density: SpinPair,
    ) -> SpinPair:
        r"""Build the Roothaan effective Fock matrix.

        1. Delegates to the base class to get :math:`(\mathbf{F}^{\alpha},
           \mathbf{F}^{\beta})`.
        2. Transforms both to the MO basis using cached coefficients.
        3. Assembles the effective Fock block-wise (Guest-Saunders).
        4. Back-transforms to the AO basis.

        Returns a shared :class:`SpinPair` containing the single
        effective Fock matrix.

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :returns: Shared effective Fock matrix pair.
        :rtype: SpinPair
        """
        # ── Physical alpha / beta Fock matrices ──────────────────────
        F_alpha, F_beta = super()._build_fock(density)
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

        c = slice(0, nc)                  # closed indices
        o = slice(nc, nc + no)            # open indices
        v = slice(nc + no, n)             # virtual indices

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

        # ── Back-transform to AO basis ───────────────────────────────
        SC = self.S @ C
        F_eff = SC @ F_eff_mo @ SC.T
        return SpinPair(F_eff)

    def _compute_electronic_energy(
        self,
        density: SpinPair,
        fock: SpinPair | tuple | None = None,
    ) -> float:
        r"""ROHF electronic energy using physical Fock matrices.

        The *fock* argument contains the effective Fock (used by the
        SCF loop for DIIS / diagonalisation) but the energy must be
        computed from the *physical* per-spin Fock matrices cached
        by :meth:`_build_fock`.

        .. math::

            E_{elec} = \tfrac{1}{2}\bigl[
                \operatorname{Tr}[\mathbf{P}^{\alpha}\,
                    (\mathbf{H} + \mathbf{F}^{\alpha})]
              + \operatorname{Tr}[\mathbf{P}^{\beta}\,
                    (\mathbf{H} + \mathbf{F}^{\beta})]\bigr]

        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param fock: Not used.  The energy is computed from the
            physical Fock matrices cached by :meth:`_build_fock`.
        :type fock: SpinPair | tuple | None
        :returns: Electronic energy (Hartree).
        :rtype: float
        """
        P_alpha, P_beta = density
        return 0.5 * (
            np.sum(P_alpha * (self.H + self._F_alpha))
            + np.sum(P_beta * (self.H + self._F_beta))
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
        """Store ROHF matrices in :attr:`matrices`.

        Keys: ``S``, ``H``, ``F_eff``, ``P``, ``F_alpha``, ``F_beta``,
        ``P_alpha``, ``P_beta``, ``C``, ``epsilon``.

        :param fock: Effective Fock matrices.
        :type fock: SpinPair
        :param density: Per-spin density matrices.
        :type density: SpinPair
        :param C: MO coefficients (identical for ROHF).
        :type C: SpinPair
        :param epsilon: Orbital energies (identical for ROHF).
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
