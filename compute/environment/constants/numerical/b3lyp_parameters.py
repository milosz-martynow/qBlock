r"""B3LYP hybrid functional parameters.

Contains the empirical mixing coefficients of the B3LYP functional,
the Becke88 exchange gradient parameter, and the Lee-Yang-Parr (LYP)
correlation parameters.

**B3LYP mixing** (Becke 1993):

.. math::

    E_{xc}^{B3LYP}
        = (1 - a_0)\,E_x^{LDA}
          + a_0\,E_x^{HF}
          + a_x\,\Delta E_x^{B88}
          + (1 - a_c)\,E_c^{VWN}
          + a_c\,E_c^{LYP}

with :math:`a_0 = 0.20`, :math:`a_x = 0.72`, :math:`a_c = 0.81`.

**Becke88 exchange** (Becke 1988):

.. math::

    \Delta E_x^{B88} = -\beta\sum_\sigma
        \int \rho_\sigma^{4/3}\,
        \frac{x_\sigma^2}{1 + 6\beta\,x_\sigma\sinh^{-1}(x_\sigma)}
        \,d\mathbf{r}

with the single empirical parameter :math:`\beta = 0.0042` (a.u.).

**LYP correlation** (Lee, Yang, Parr 1988):

The four parameters :math:`a`, :math:`b`, :math:`c`, :math:`d` were
fitted by Lee, Yang, and Parr to the helium atom correlation energy
and the gradient expansion of the correlation hole.

References
----------
A. D. Becke,
*A new mixing of Hartree-Fock and local density-functional theories*,
J. Chem. Phys. **98**, 1372 (1993).
https://doi.org/10.1063/1.464304

A. D. Becke,
*Density-functional exchange-energy approximation with correct
asymptotic behavior*,
Phys. Rev. A **38**, 3098 (1988).
https://doi.org/10.1103/PhysRevA.38.3098

C. Lee, W. Yang, R. G. Parr,
*Development of the Colle-Salvetti correlation-energy formula into a
functional of the electron density*,
Phys. Rev. B **37**, 785 (1988).
https://doi.org/10.1103/PhysRevB.37.785

Constants
---------
B3LYP_A0
    Exact (HF) exchange mixing parameter :math:`a_0 = 0.20`.
B3LYP_AX
    Becke88 gradient exchange correction weight :math:`a_x = 0.72`.
B3LYP_AC
    LYP correlation weight :math:`a_c = 0.81`.
B3LYP_BETA_B88
    Becke88 empirical parameter :math:`\beta = 0.0042` (a.u.).
LYP_A, LYP_B, LYP_C, LYP_D
    Lee-Yang-Parr correlation parameters.
"""

# ──────────────────────────────────────────────────────────────────────
# B3LYP mixing parameters (Becke 1993)
# ──────────────────────────────────────────────────────────────────────
B3LYP_A0: float = 0.20   # Exact exchange fraction
B3LYP_AX: float = 0.72   # Becke88 gradient correction weight
B3LYP_AC: float = 0.81   # LYP correlation weight

# ──────────────────────────────────────────────────────────────────────
# Becke88 exchange gradient parameter (Becke 1988)
# ──────────────────────────────────────────────────────────────────────
B3LYP_BETA_B88: float = 0.0042  # Fitted to noble gas exchange energies

# ──────────────────────────────────────────────────────────────────────
# Lee-Yang-Parr correlation parameters (Lee, Yang, Parr 1988)
# ──────────────────────────────────────────────────────────────────────
LYP_A: float = 0.04918
LYP_B: float = 0.132
LYP_C: float = 0.2533
LYP_D: float = 0.349
