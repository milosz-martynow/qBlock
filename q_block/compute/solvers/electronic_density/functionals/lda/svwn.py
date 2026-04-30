r"""Slater-Vosko-Wilk-Nusair (SVWN) LDA functional.

Implements the Local Density Approximation (LDA) exchange-correlation
functional combining Slater (Dirac) exchange with Vosko-Wilk-Nusair
(VWN5) correlation.

**Exchange** (Slater/Dirac):

.. math::

    \varepsilon_x^{LDA}(\rho)
        = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}
          \rho^{1/3}

**Correlation** (VWN5 parametrisation of the homogeneous electron gas):

The VWN5 correlation energy per electron :math:`\varepsilon_c(r_s)`
is parametrised in terms of the Wigner-Seitz radius
:math:`r_s = (3 / 4\pi\rho)^{1/3}` using the Padé-like form
given by Vosko, Wilk, and Nusair (1980).

Classes
-------
SVWN
    Slater-VWN5 LDA functional.
"""

import logging
from typing import Tuple

import numpy as np

from q_block.compute.environment.constants.numerical import (
    VWN_A_F,
    VWN_A_P,
    VWN_B_F,
    VWN_B_P,
    VWN_C_F,
    VWN_C_F_SPIN,
    VWN_C_P,
    VWN_X0_F,
    VWN_X0_P,
)
from q_block.compute.solvers.electronic_density.functionals.exchange_correlation_functional import (
    ExchangeCorrelationFunctional,
)

logger = logging.getLogger(__name__)


def _vwn5_epsilon_c(
    rs: np.ndarray,
    A: float,
    x0: float,
    b: float,
    c: float,
) -> np.ndarray:
    r"""VWN5 correlation energy per electron.

    The Padé approximant form:

    .. math::

        \varepsilon_c(r_s) = \frac{A}{2}
            \left[
                \ln\frac{x^2}{X(x)}
                + \frac{2b}{Q}\arctan\frac{Q}{2x + b}
                - \frac{bx_0}{X(x_0)}
                \left(
                    \ln\frac{(x - x_0)^2}{X(x)}
                    + \frac{2(b + 2x_0)}{Q}\arctan\frac{Q}{2x + b}
                \right)
            \right]

    where :math:`x = \sqrt{r_s}`, :math:`X(x) = x^2 + bx + c`,
    and :math:`Q = \sqrt{4c - b^2}`.

    :param rs: Wigner-Seitz radii.
    :type rs: np.ndarray
    :param A: VWN parameter A.
    :type A: float
    :param x0: VWN parameter x_0.
    :type x0: float
    :param b: VWN parameter b.
    :type b: float
    :param c: VWN parameter c.
    :type c: float
    :returns: Correlation energy per electron.
    :rtype: np.ndarray
    """
    x = np.sqrt(rs)
    X_x = x * x + b * x + c
    X_x0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4.0 * c - b * b)

    term1 = np.log(x * x / X_x)
    term2 = (2.0 * b / Q) * np.arctan(Q / (2.0 * x + b))
    term3_log = np.log((x - x0) ** 2 / X_x)
    term3_atan = (
        (2.0 * (b + 2.0 * x0) / Q)
        * np.arctan(Q / (2.0 * x + b))
    )
    term3 = (b * x0 / X_x0) * (term3_log + term3_atan)

    return (A / 2.0) * (term1 + term2 - term3)


def _vwn5_d_epsilon_c(
    rs: np.ndarray,
    A: float,
    x0: float,
    b: float,
    c: float,
) -> np.ndarray:
    r"""Derivative of VWN5 :math:`\varepsilon_c` w.r.t. :math:`r_s`.

    Computed analytically for the VWN Padé form.

    :param rs: Wigner-Seitz radii.
    :type rs: np.ndarray
    :param A: VWN parameter A.
    :type A: float
    :param x0: VWN parameter x_0.
    :type x0: float
    :param b: VWN parameter b.
    :type b: float
    :param c: VWN parameter c.
    :type c: float
    :returns: :math:`d\varepsilon_c / dr_s`.
    :rtype: np.ndarray
    """
    x = np.sqrt(rs)
    X_x = x * x + b * x + c
    X_x0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4.0 * c - b * b)
    dx_drs = 0.5 / x

    # d/dx of each term
    dX_dx = 2.0 * x + b
    d_term1 = 2.0 / x - dX_dx / X_x
    d_term2 = (
        (2.0 * b / Q)
        * (-2.0 / ((2.0 * x + b) ** 2 + Q * Q))
        * 2.0
    )
    d_term3_log = 2.0 / (x - x0) - dX_dx / X_x
    d_term3_atan = (
        (2.0 * (b + 2.0 * x0) / Q)
        * (-2.0 / ((2.0 * x + b) ** 2 + Q * Q))
        * 2.0
    )
    d_term3 = (b * x0 / X_x0) * (
        d_term3_log + d_term3_atan
    )

    d_ec_dx = (A / 2.0) * (d_term1 + d_term2 - d_term3)
    return d_ec_dx * dx_drs


class SVWN(ExchangeCorrelationFunctional):
    r"""Slater-VWN5 Local Density Approximation functional.

    Combines Slater (Dirac) exchange with VWN5 correlation for the
    homogeneous electron gas.

    For spin-polarised systems the spin-scaling relations are used:

    .. math::

        \varepsilon_{xc}(\rho_\alpha, \rho_\beta)
            = \varepsilon_{xc}^P(\rho)\,
              [1 - f(\zeta)]
            + \varepsilon_{xc}^F(\rho)\,f(\zeta)

    where :math:`\zeta = (\rho_\alpha - \rho_\beta)/\rho` is the
    spin polarisation and :math:`f(\zeta)` is the spin interpolation
    function.
    """

    def __init__(self) -> None:
        super().__init__(
            name="SVWN",
            functional_type="lda",
            exact_exchange_fraction=0.0,
        )

    def compute_exc_vxc(
        self,
        rho_alpha: np.ndarray,
        rho_beta: np.ndarray,
        **_kwargs: object,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Compute SVWN energy density and potentials.

        :param rho_alpha: Alpha density, shape ``(n_points,)``.
        :type rho_alpha: np.ndarray
        :param rho_beta: Beta density, shape ``(n_points,)``.
        :type rho_beta: np.ndarray
        :returns: ``(exc, vxc_alpha, vxc_beta)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        rho = rho_alpha + rho_beta
        n_pts = len(rho)
        exc = np.zeros(n_pts)
        vxc_alpha = np.zeros(n_pts)
        vxc_beta = np.zeros(n_pts)

        # Mask out negligible density
        mask = rho > 1e-18
        rho_m = rho[mask]
        rho_a_m = rho_alpha[mask]
        rho_b_m = rho_beta[mask]

        # ── Exchange (Slater/Dirac) ──────────────────────────────
        cx = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
        # Spin-scaled exchange: E_x = cx * 2^(1/3) * (rho_a^(4/3) + rho_b^(4/3))
        scale = 2.0 ** (1.0 / 3.0)
        ex_a = cx * scale * rho_a_m ** (4.0 / 3.0)
        ex_b = cx * scale * rho_b_m ** (4.0 / 3.0)
        exc_x = (ex_a + ex_b) / rho_m

        # Exchange potential v_x^sigma = (4/3) * cx * 2^(1/3) * rho_sigma^(1/3)
        vx_a = (4.0 / 3.0) * cx * scale * rho_a_m ** (
            1.0 / 3.0
        )
        vx_b = (4.0 / 3.0) * cx * scale * rho_b_m ** (
            1.0 / 3.0
        )

        # ── Correlation (VWN5) ───────────────────────────────────
        rs = (3.0 / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)

        # Paramagnetic correlation
        ec_p = _vwn5_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )
        dec_p = _vwn5_d_epsilon_c(
            rs, VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
        )

        # Ferromagnetic correlation
        ec_f = _vwn5_epsilon_c(
            rs, VWN_A_F, VWN_X0_F, VWN_B_F, VWN_C_F
        )
        dec_f = _vwn5_d_epsilon_c(
            rs, VWN_A_F, VWN_X0_F, VWN_B_F, VWN_C_F
        )

        # Spin polarisation and interpolation
        zeta = (rho_a_m - rho_b_m) / rho_m
        zeta = np.clip(zeta, -1.0 + 1e-15, 1.0 - 1e-15)

        f_zeta = (
            (1.0 + zeta) ** (4.0 / 3.0)
            + (1.0 - zeta) ** (4.0 / 3.0)
            - 2.0
        ) / (2.0 * VWN_C_F_SPIN)

        df_zeta = (
            (4.0 / 3.0)
            * (
                (1.0 + zeta) ** (1.0 / 3.0)
                - (1.0 - zeta) ** (1.0 / 3.0)
            )
            / (2.0 * VWN_C_F_SPIN)
        )

        exc_c = ec_p + f_zeta * (ec_f - ec_p)

        # Correlation potential:
        # v_c^sigma = ec + rho * d(ec)/d(rho)
        # = ec - (rs/3) * d(ec)/d(rs)
        # + spin contribution
        drs_drho = -(rs / (3.0 * rho_m))
        dec_drho = (dec_p + f_zeta * (dec_f - dec_p)) * drs_drho

        # dzeta/drho_alpha = (1 - zeta) / rho,
        # dzeta/drho_beta = -(1 + zeta) / rho
        dzeta_drho_a = (1.0 - zeta) / rho_m
        dzeta_drho_b = -(1.0 + zeta) / rho_m

        dec_dfzeta = (ec_f - ec_p) * df_zeta

        vc_a = exc_c + rho_m * dec_drho + rho_m * dec_dfzeta * dzeta_drho_a
        vc_b = exc_c + rho_m * dec_drho + rho_m * dec_dfzeta * dzeta_drho_b

        # ── Total ────────────────────────────────────────────────
        exc[mask] = exc_x + exc_c
        vxc_alpha[mask] = vx_a + vc_a
        vxc_beta[mask] = vx_b + vc_b

        return exc, vxc_alpha, vxc_beta
