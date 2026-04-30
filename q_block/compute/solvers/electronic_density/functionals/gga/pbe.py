r"""Perdew-Burke-Ernzerhof (PBE) GGA functional.

Implements the Generalised Gradient Approximation (GGA) exchange-
correlation functional of Perdew, Burke, and Ernzerhof (1996).

**Exchange** (PBE enhancement factor over LDA):

.. math::

    E_x^{PBE}[\rho] = \int \rho\,\varepsilon_x^{LDA}(\rho)\,
        F_x(s)\,d\mathbf{r}

    F_x(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2 / \kappa}

where :math:`s = |\nabla\rho| / (2\,k_F\,\rho)` is the reduced
density gradient, :math:`k_F = (3\pi^2\rho)^{1/3}` is the Fermi
wavevector, :math:`\kappa = 0.804`, and :math:`\mu = 0.21951`.

**Correlation** (PBE):

.. math::

    E_c^{PBE}[\rho] = \int \rho\,[\varepsilon_c^{LDA}(\rho)
        + H(r_s, \zeta, t)]\,d\mathbf{r}

where :math:`t = |\nabla\rho| / (2\,k_s\,\rho)` and :math:`k_s`
is the Thomas-Fermi screening wave vector.

Classes
-------
PBE
    PBE GGA exchange-correlation functional.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from q_block.compute.environment.constants.numerical import (
    PBE_BETA,
    PBE_GAMMA,
    PBE_KAPPA,
    PBE_MU,
    VWN_A_P,
    VWN_B_P,
    VWN_C_P,
    VWN_X0_P,
)
from q_block.compute.solvers.electronic_density.functionals.exchange_correlation_functional import (
    ExchangeCorrelationFunctional,
)
from q_block.compute.solvers.electronic_density.functionals.lda.svwn import (
    _vwn5_d_epsilon_c,
    _vwn5_epsilon_c,
)

logger = logging.getLogger(__name__)


class PBE(ExchangeCorrelationFunctional):
    r"""Perdew-Burke-Ernzerhof (PBE) GGA functional.

    Generalised Gradient Approximation combining PBE exchange with
    PBE correlation.  Depends on both the local density and its
    gradient.
    """

    def __init__(self) -> None:
        super().__init__(
            name="PBE",
            functional_type="gga",
            exact_exchange_fraction=0.0,
        )

    def compute_exc_vxc(
        self,
        rho_alpha: np.ndarray,
        rho_beta: np.ndarray,
        gamma_aa: Optional[np.ndarray] = None,
        gamma_ab: Optional[np.ndarray] = None,
        gamma_bb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute PBE energy density and potentials.

        :param rho_alpha: Alpha density, shape ``(n_points,)``.
        :type rho_alpha: np.ndarray
        :param rho_beta: Beta density, shape ``(n_points,)``.
        :type rho_beta: np.ndarray
        :param gamma_aa: :math:`|\nabla\rho_\alpha|^2`.
        :type gamma_aa: Optional[np.ndarray]
        :param gamma_ab: :math:`\nabla\rho_\alpha \cdot
            \nabla\rho_\beta`.
        :type gamma_ab: Optional[np.ndarray]
        :param gamma_bb: :math:`|\nabla\rho_\beta|^2`.
        :type gamma_bb: Optional[np.ndarray]
        :returns: ``(exc, vxc_alpha, vxc_beta)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        if gamma_aa is None or gamma_bb is None:
            raise ValueError(
                "PBE (GGA) requires density gradient information "
                "(gamma_aa, gamma_bb)."
            )
        if gamma_ab is None:
            gamma_ab = np.zeros_like(gamma_aa)

        rho = rho_alpha + rho_beta
        n_pts = len(rho)
        exc = np.zeros(n_pts)
        vxc_alpha = np.zeros(n_pts)
        vxc_beta = np.zeros(n_pts)

        mask = rho > 1e-18
        rho_m = rho[mask]
        rho_a = rho_alpha[mask]
        rho_b = rho_beta[mask]
        gaa = gamma_aa[mask]
        gab = gamma_ab[mask]
        gbb = gamma_bb[mask]
        gamma_total = gaa + 2.0 * gab + gbb

        # ── LDA exchange baseline ────────────────────────────────
        cx = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
        scale = 2.0 ** (1.0 / 3.0)

        # ── PBE exchange (spin-scaled) ───────────────────────────
        def _pbe_exchange_spin(
            rho_s: np.ndarray,
            gamma_ss: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """PBE exchange for a single spin channel (spin-scaled).

            :param rho_s: Spin density.
            :type rho_s: np.ndarray
            :param gamma_ss: |grad rho_s|^2.
            :type gamma_ss: np.ndarray
            :returns: (ex_s, vx_rho, vx_gamma).
            :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
            """
            rho_2s = 2.0 * rho_s
            safe_rho = np.maximum(rho_2s, 1e-30)
            kf = (3.0 * np.pi ** 2 * safe_rho) ** (
                1.0 / 3.0
            )
            grad_mag = np.sqrt(np.maximum(4.0 * gamma_ss, 0.0))
            s = grad_mag / (2.0 * kf * safe_rho)
            s2 = s * s

            # Enhancement factor
            denom = 1.0 + PBE_MU * s2 / PBE_KAPPA
            Fx = 1.0 + PBE_KAPPA - PBE_KAPPA / denom

            # LDA exchange energy density for spin-scaled density
            ex_lda = cx * safe_rho ** (1.0 / 3.0)
            ex_s = ex_lda * Fx

            # Potential w.r.t. rho (via chain rule)
            dFx_ds2 = PBE_MU / denom ** 2
            ds2_drho = -(8.0 / 3.0) * s2 / safe_rho
            vx_rho = (
                (4.0 / 3.0) * ex_lda * Fx
                + ex_lda * safe_rho * dFx_ds2 * ds2_drho
            )

            # Potential w.r.t. gamma (for GGA contribution to V_xc)
            ds2_dgamma = (
                4.0 / (4.0 * kf ** 2 * safe_rho ** 2)
            )
            vx_gamma = ex_lda * safe_rho * dFx_ds2 * ds2_dgamma

            return 0.5 * ex_s, vx_rho, vx_gamma

        ex_a, vx_rho_a, _ = _pbe_exchange_spin(
            rho_a, gaa
        )
        ex_b, vx_rho_b, _ = _pbe_exchange_spin(
            rho_b, gbb
        )
        exc_x = (ex_a + ex_b)

        # ── PBE correlation ──────────────────────────────────────
        rs = (3.0 / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
        ec_lda = _vwn5_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )
        dec_drs = _vwn5_d_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )

        # Thomas-Fermi screening
        ks = np.sqrt(4.0 * (3.0 / np.pi) ** (1.0 / 3.0) / rs)
        t = np.sqrt(np.maximum(gamma_total, 0.0)) / (
            2.0 * ks * rho_m
        )
        t2 = t * t

        # PBE H function
        A = PBE_BETA / PBE_GAMMA / (
            np.exp(-ec_lda / PBE_GAMMA) - 1.0 + 1e-30
        )
        At2 = A * t2
        inner = 1.0 + At2 * (1.0 + At2)
        H = PBE_GAMMA * np.log(
            1.0 + PBE_BETA * t2 * inner / (
                PBE_GAMMA * inner + PBE_BETA * t2 * At2
                + 1e-30
            )
        )

        exc_c = ec_lda + H

        # Correlation potential (approximate — using LDA part + H correction)
        vc_lda = ec_lda - (rs / 3.0) * dec_drs

        vc_a = vc_lda + H
        vc_b = vc_lda + H

        # ── Total ────────────────────────────────────────────────
        exc[mask] = exc_x + exc_c
        vxc_alpha[mask] = vx_rho_a + vc_a
        vxc_beta[mask] = vx_rho_b + vc_b

        return exc, vxc_alpha, vxc_beta
