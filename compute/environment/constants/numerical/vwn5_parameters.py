r"""Vosko-Wilk-Nusair (VWN5) LDA correlation parameters.

Contains the fitted parameters of the VWN5 Padé approximant for the
correlation energy of the homogeneous electron gas, as published by
Vosko, Wilk, and Nusair (1980), Table 5 (parametrisation V).

The correlation energy per electron is expressed in terms of the
Wigner-Seitz radius :math:`r_s = (3 / 4\pi\rho)^{1/3}` via:

.. math::

    \varepsilon_c(r_s) = \frac{A}{2}
        \left[
            \ln\frac{x^2}{X(x)}
            + \frac{2b}{Q}\arctan\frac{Q}{2x + b}
            - \frac{bx_0}{X(x_0)}
            \left(
                \ln\frac{(x - x_0)^2}{X(x)}
                + \frac{2(b + 2x_0)}{Q}
                  \arctan\frac{Q}{2x + b}
            \right)
        \right]

where :math:`x = \sqrt{r_s}`, :math:`X(t) = t^2 + bt + c`, and
:math:`Q = \sqrt{4c - b^2}`.

Two limit cases are parametrised:

* **Paramagnetic** (P) — fully unpolarised electron gas
  (:math:`\zeta = 0`).
* **Ferromagnetic** (F) — fully polarised electron gas
  (:math:`\zeta = 1`).

The spin-interpolation constant :math:`c_f = 2^{1/3} - 1` appears in
the von Barth–Hedin interpolation function

.. math::

    f(\zeta) = \frac{(1+\zeta)^{4/3} + (1-\zeta)^{4/3} - 2}
                    {2\,c_f}

used to interpolate between the paramagnetic and ferromagnetic limits.

References
----------
S. H. Vosko, L. Wilk, M. Nusair,
*Accurate spin-dependent electron liquid correlation energies for
local spin density calculations: a critical analysis*,
Can. J. Phys. **58**, 1200–1211 (1980).
https://doi.org/10.1139/p80-159  (see Table 5, parametrisation V)

Constants
---------
VWN_A_P, VWN_X0_P, VWN_B_P, VWN_C_P
    Paramagnetic Padé parameters :math:`A`, :math:`x_0`, :math:`b`,
    :math:`c`.
VWN_A_F, VWN_X0_F, VWN_B_F, VWN_C_F
    Ferromagnetic Padé parameters :math:`A`, :math:`x_0`, :math:`b`,
    :math:`c`.
VWN_C_F_SPIN
    Spin-interpolation prefactor :math:`c_f = 2^{1/3} - 1 \approx
    0.2599`.
"""

# ──────────────────────────────────────────────────────────────────────
# Paramagnetic (ζ = 0) VWN5 parameters — Table 5, column "ε_c^P"
# ──────────────────────────────────────────────────────────────────────
VWN_A_P: float = 0.0621814
VWN_X0_P: float = -0.10498
VWN_B_P: float = 3.72744
VWN_C_P: float = 12.9352

# ──────────────────────────────────────────────────────────────────────
# Ferromagnetic (ζ = 1) VWN5 parameters — Table 5, column "ε_c^F"
# ──────────────────────────────────────────────────────────────────────
VWN_A_F: float = 0.0310907
VWN_X0_F: float = -0.32500
VWN_B_F: float = 7.06042
VWN_C_F: float = 18.0578

# ──────────────────────────────────────────────────────────────────────
# Spin-interpolation constant  c_f = 2^(1/3) − 1 ≈ 0.2599
# Derived analytically from the von Barth–Hedin spin-scaling relation.
# ──────────────────────────────────────────────────────────────────────
VWN_C_F_SPIN: float = 2.0 ** (1.0 / 3.0) - 1.0
