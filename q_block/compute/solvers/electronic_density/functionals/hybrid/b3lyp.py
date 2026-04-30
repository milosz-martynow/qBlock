r"""Becke three-parameter Lee-Yang-Parr (B3LYP) hybrid functional.

Implements the B3LYP hybrid exchange-correlation functional, which
mixes exact (Hartree-Fock) exchange with DFT exchange and correlation:

.. math::

    E_{xc}^{B3LYP}
        = (1 - a_0)\,E_x^{LDA}
          + a_0\,E_x^{HF}
          + a_x\,\Delta E_x^{B88}
          + (1 - a_c)\,E_c^{VWN}
          + a_c\,E_c^{LYP}

with Becke's original parameters:

* :math:`a_0 = 0.20` (exact exchange)
* :math:`a_x = 0.72` (Becke88 gradient exchange correction)
* :math:`a_c = 0.81` (LYP correlation)

**Exchange components:**

* Slater (Dirac) LDA exchange
* Becke88 gradient correction :math:`\Delta E_x^{B88}`

**Correlation components:**

* VWN5 LDA correlation
* Lee-Yang-Parr (LYP) GGA correlation

Classes
-------
B3LYP
    B3LYP hybrid exchange-correlation functional.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from q_block.compute.environment.constants.numerical import (
    B3LYP_A0,
    B3LYP_AC,
    B3LYP_AX,
    B3LYP_BETA_B88,
    LYP_A,
    LYP_B,
    LYP_C,
    LYP_D,
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


class B3LYP(ExchangeCorrelationFunctional):
    r"""B3LYP hybrid exchange-correlation functional.

    Three-parameter hybrid mixing exact (HF) exchange with
    DFT exchange (Slater + Becke88) and correlation (VWN5 + LYP).

    The exact exchange fraction is :math:`a_0 = 0.20`, which means
    20% of the Coulomb exchange matrix from ERIs is included in the
    Kohn-Sham Fock matrix.

    Attributes
    ----------
    a0 : float
        Exact exchange mixing parameter.
    ax : float
        Becke88 exchange gradient correction weight.
    ac : float
        LYP correlation weight.
    """

    def __init__(self) -> None:
        super().__init__(
            name="B3LYP",
            functional_type="hybrid",
            exact_exchange_fraction=B3LYP_A0,
        )
        self.a0: float = B3LYP_A0
        self.ax: float = B3LYP_AX
        self.ac: float = B3LYP_AC

    @staticmethod
    def _becke88_exchange(
        rho_sigma: np.ndarray,
        gamma_ss: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Becke88 exchange energy density and derivatives for one spin.

        Computes the Becke88 gradient correction per unit volume:

        .. math::

            f^{B88}_\sigma
                = -\beta\,\rho_\sigma^{4/3}\,
                  \frac{x_\sigma^2}{1 + 6\beta\,x_\sigma\,
                  \sinh^{-1}(x_\sigma)}

        where :math:`x_\sigma = |\nabla\rho_\sigma| /
        \rho_\sigma^{4/3}`.

        :param rho_sigma: Per-spin density :math:`\rho_\sigma`.
        :type rho_sigma: np.ndarray
        :param gamma_ss: :math:`|\nabla\rho_\sigma|^2`.
        :type gamma_ss: np.ndarray
        :returns: ``(f_b88, df_drho, df_dgamma)`` — B88 energy density
            per volume, :math:`\partial f / \partial\rho_\sigma` and
            :math:`\partial f / \partial\gamma_{\sigma\sigma}`.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        n_pts = len(rho_sigma)
        f_b88 = np.zeros(n_pts)
        df_drho = np.zeros(n_pts)
        df_dgamma = np.zeros(n_pts)

        mask = rho_sigma > 1e-15
        rho = rho_sigma[mask]
        rho_13 = rho ** (1.0 / 3.0)
        rho_43 = rho * rho_13
        grad_mag = np.sqrt(np.maximum(gamma_ss[mask], 0.0))
        x = grad_mag / rho_43
        x2 = x * x
        asinh_x = np.arcsinh(x)
        denom = 1.0 + 6.0 * B3LYP_BETA_B88 * x * asinh_x

        # f = -beta * rho^{4/3} * g(x)  where g(x) = x^2 / denom
        g = x2 / denom
        f_b88[mask] = -B3LYP_BETA_B88 * rho_43 * g

        # df/drho = -beta * (4/3) * rho^{1/3} * g
        #         + (-beta * rho^{4/3}) * dg/dx * dx/drho
        ddenom_dx = 6.0 * B3LYP_BETA_B88 * (
            asinh_x + x / np.sqrt(1.0 + x2)
        )
        dg_dx = (2.0 * x * denom - x2 * ddenom_dx) / denom ** 2
        dx_drho = -(4.0 / 3.0) * x / rho

        df_drho[mask] = -B3LYP_BETA_B88 * (
            (4.0 / 3.0) * rho_13 * g + rho_43 * dg_dx * dx_drho
        )

        # df/dgamma = -beta * rho^{4/3} * dg/dx * dx/dgamma
        # x = gamma^{1/2} / rho^{4/3}
        # dx/dgamma = 1 / (2 * gamma^{1/2} * rho^{4/3})
        dx_dgamma = np.zeros_like(rho)
        grad_ok = grad_mag > 1e-30
        dx_dgamma[grad_ok] = 0.5 / (
            grad_mag[grad_ok] * rho_43[grad_ok]
        )
        df_dgamma[mask] = (
            -B3LYP_BETA_B88 * rho_43 * dg_dx * dx_dgamma
        )

        return f_b88, df_drho, df_dgamma

    @staticmethod
    def _lyp_correlation(
        rho: np.ndarray,
        rho_a: np.ndarray,
        rho_b: np.ndarray,
        gamma_total: np.ndarray,
        gamma_aa: np.ndarray,
        gamma_bb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Lee-Yang-Parr correlation energy density and potentials.

        :param rho: Total density.
        :type rho: np.ndarray
        :param rho_a: Alpha density.
        :type rho_a: np.ndarray
        :param rho_b: Beta density.
        :type rho_b: np.ndarray
        :param gamma_total: :math:`|\nabla\rho|^2`.
        :type gamma_total: np.ndarray
        :param gamma_aa: :math:`|\nabla\rho_\alpha|^2`.
        :type gamma_aa: np.ndarray
        :param gamma_bb: :math:`|\nabla\rho_\beta|^2`.
        :type gamma_bb: np.ndarray
        :returns: ``(ec_lyp, vc_a, vc_b)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        a, b, c, d = LYP_A, LYP_B, LYP_C, LYP_D
        safe_rho = np.maximum(rho, 1e-30)
        rho_m13 = safe_rho ** (-1.0 / 3.0)

        omega = np.exp(-c * rho_m13) / (
            1.0 + d * rho_m13
        ) * safe_rho ** (-11.0 / 3.0)
        delta = c * rho_m13 + d * rho_m13 / (
            1.0 + d * rho_m13
        )
        CF = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0)

        rho_a_83 = np.maximum(rho_a, 1e-30) ** (8.0 / 3.0)
        rho_b_83 = np.maximum(rho_b, 1e-30) ** (8.0 / 3.0)

        # LYP energy density
        term1 = -a * (
            4.0 * rho_a * rho_b / (safe_rho * (1.0 + d * rho_m13))
        )
        term2 = -a * b * omega * rho_a * rho_b * (
            2.0 ** (11.0 / 3.0) * CF * (rho_a_83 + rho_b_83)
            + (47.0 / 18.0 - 7.0 * delta / 18.0) * gamma_total
            - (5.0 / 2.0 - delta / 18.0) * (gamma_aa + gamma_bb)
            - (delta - 11.0) / 9.0 * (
                rho_a * gamma_aa / (np.maximum(rho_a, 1e-30))
                + rho_b * gamma_bb / (np.maximum(rho_b, 1e-30))
            )
        )

        ec_lyp = (term1 + term2) / safe_rho

        # Approximate potentials
        vc_a = ec_lyp
        vc_b = ec_lyp

        return ec_lyp, vc_a, vc_b

    def compute_exc_vxc(
        self,
        rho_alpha: np.ndarray,
        rho_beta: np.ndarray,
        gamma_aa: Optional[np.ndarray] = None,
        gamma_ab: Optional[np.ndarray] = None,
        gamma_bb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute B3LYP energy density and potentials.

        Note: the exact exchange contribution is **not** included here;
        it is handled by the Kohn-Sham solver via the ERI tensor and
        the :attr:`exact_exchange_fraction`.

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
                "B3LYP (hybrid) requires density gradient "
                "information (gamma_aa, gamma_bb)."
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

        # ── LDA exchange (Slater) ────────────────────────────────
        cx = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
        scale = 2.0 ** (1.0 / 3.0)

        ex_lda_a = cx * scale * rho_a ** (4.0 / 3.0)
        ex_lda_b = cx * scale * rho_b ** (4.0 / 3.0)
        exc_lda_x = (ex_lda_a + ex_lda_b) / rho_m

        vx_lda_a = (4.0 / 3.0) * cx * scale * np.maximum(
            rho_a, 1e-30
        ) ** (1.0 / 3.0)
        vx_lda_b = (4.0 / 3.0) * cx * scale * np.maximum(
            rho_b, 1e-30
        ) ** (1.0 / 3.0)

        # ── Becke88 exchange correction ──────────────────────────
        f_b88_a, df_a_drho, _ = self._becke88_exchange(
            rho_a, gaa
        )
        f_b88_b, df_b_drho, _ = self._becke88_exchange(
            rho_b, gbb
        )
        exc_b88 = (f_b88_a + f_b88_b) / rho_m

        # ── VWN5 correlation ─────────────────────────────────────
        rs = (3.0 / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
        ec_vwn = _vwn5_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )
        dec_vwn = _vwn5_d_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )
        vc_vwn = ec_vwn - (rs / 3.0) * dec_vwn

        # ── LYP correlation ──────────────────────────────────────
        ec_lyp, vc_lyp_a, vc_lyp_b = self._lyp_correlation(
            rho_m, rho_a, rho_b, gamma_total, gaa, gbb
        )

        # ── B3LYP mixing ────────────────────────────────────────
        # E_xc = (1 - a0) * E_x^LDA + a_x * dE_x^B88
        #      + (1 - a_c) * E_c^VWN + a_c * E_c^LYP
        # The a0 * E_x^HF part is handled by the KS solver
        exc_dft_x = (
            (1.0 - self.a0) * exc_lda_x + self.ax * exc_b88
        )
        exc_dft_c = (1.0 - self.ac) * ec_vwn + self.ac * ec_lyp

        exc[mask] = exc_dft_x + exc_dft_c

        vxc_alpha[mask] = (
            (1.0 - self.a0) * vx_lda_a
            + self.ax * df_a_drho
            + (1.0 - self.ac) * vc_vwn
            + self.ac * vc_lyp_a
        )
        vxc_beta[mask] = (
            (1.0 - self.a0) * vx_lda_b
            + self.ax * df_b_drho
            + (1.0 - self.ac) * vc_vwn
            + self.ac * vc_lyp_b
        )

        return exc, vxc_alpha, vxc_beta
