r"""Perdew-Burke-Ernzerhof (PBE) GGA functional parameters.

Contains the universal parameters of the PBE exchange-correlation
functional as defined by Perdew, Burke, and Ernzerhof (1996).

**Exchange** parameters appear in the PBE enhancement factor:

.. math::

    F_x(s) = 1 + \kappa - \frac{\kappa}{1 + \mu\,s^2 / \kappa}

where :math:`s = |\nabla\rho| / (2\,k_F\,\rho)` is the reduced
density gradient.  The values :math:`\kappa = 0.804` and
:math:`\mu = 0.21951` satisfy the local Lieb–Oxford bound and recover
the correct linear response of the uniform electron gas, respectively.

**Correlation** parameters appear in the PBE :math:`H` function:

.. math::

    H = \gamma \ln\!\left(
            1 + \beta\,t^2\,\frac{1 + A\,t^2}{1 + A\,t^2 + A^2\,t^4}
        \right)

where :math:`t = |\nabla\rho| / (2\,k_s\,\rho)` and
:math:`k_s` is the Thomas-Fermi screening wave vector.
The relationship :math:`\gamma = (1 - \ln 2)/\pi^2 \approx 0.031091`
ensures recovery of the second-order gradient expansion for
correlation.

References
----------
J. P. Perdew, K. Burke, M. Ernzerhof,
*Generalized Gradient Approximation Made Simple*,
Phys. Rev. Lett. **77**, 3865 (1996).
https://doi.org/10.1103/PhysRevLett.77.3865

Constants
---------
PBE_KAPPA
    Exchange enhancement factor parameter :math:`\kappa = 0.804`.
PBE_MU
    Exchange enhancement factor parameter :math:`\mu = 0.21951`.
PBE_GAMMA
    Correlation :math:`\gamma = (1 - \ln 2)/\pi^2 \approx 0.031091`.
PBE_BETA
    Correlation :math:`\beta = 0.066725`.
"""

# ──────────────────────────────────────────────────────────────────────
# PBE exchange parameters
# ──────────────────────────────────────────────────────────────────────
PBE_KAPPA: float = 0.804    # Lieb–Oxford bound parameter
PBE_MU: float = 0.21951     # Second-order gradient expansion constant

# ──────────────────────────────────────────────────────────────────────
# PBE correlation parameters
# ──────────────────────────────────────────────────────────────────────
PBE_GAMMA: float = 0.031091  # (1 − ln 2) / π²
PBE_BETA: float = 0.066725   # Correlation H-function parameter
