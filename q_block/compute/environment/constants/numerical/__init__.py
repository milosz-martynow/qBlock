"""Numerical data sets (basis sets, atomic radii, quadratures, etc.)."""

from .b3lyp_parameters import (
    B3LYP_A0,
    B3LYP_AC,
    B3LYP_AX,
    B3LYP_BETA_B88,
    LYP_A,
    LYP_B,
    LYP_C,
    LYP_D,
)
from .bragg_slater_radii import BRAGG_SLATER_RADII
from .lebedev_quadrature import (
    LEBEDEV_GRIDS,
    lebedev_6,
    lebedev_14,
    lebedev_26,
)
from .pbe_parameters import (
    PBE_BETA,
    PBE_GAMMA,
    PBE_KAPPA,
    PBE_MU,
)
from .vwn5_parameters import (
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

__all__ = [
    # B3LYP / Becke88 / LYP
    "B3LYP_A0",
    "B3LYP_AC",
    "B3LYP_AX",
    "B3LYP_BETA_B88",
    "LYP_A",
    "LYP_B",
    "LYP_C",
    "LYP_D",
    # Bragg-Slater / Lebedev
    "BRAGG_SLATER_RADII",
    "LEBEDEV_GRIDS",
    "lebedev_6",
    "lebedev_14",
    "lebedev_26",
    # PBE
    "PBE_BETA",
    "PBE_GAMMA",
    "PBE_KAPPA",
    "PBE_MU",
    # VWN5
    "VWN_A_F",
    "VWN_A_P",
    "VWN_B_F",
    "VWN_B_P",
    "VWN_C_F",
    "VWN_C_F_SPIN",
    "VWN_C_P",
    "VWN_X0_F",
    "VWN_X0_P",
]
