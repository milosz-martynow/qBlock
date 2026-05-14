"""Shared constants and configuration for qBlock test suite.

This module provides common paths, basis set configurations, and test values
used across multiple test modules to ensure consistency and reduce duplication.
"""

from pathlib import Path
from typing import List

from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.coordinates import CartesianCoordinates

# ======================================================================
# Path Constants
# ======================================================================

BASIS_ROOT: Path = Path("./q_block/compute/environment/constants/numerical/basis_set/pople")
"""Root directory for Gaussian-format (GTO) basis set files."""

STO_ROOT: Path = Path("./q_block/compute/environment/constants/numerical/basis_set/pople")
"""Root directory for Slater-type (STO) basis set files."""

GOLDEN_ROOT: Path = Path("./q_block/tests/verification/verification_data/gto_population")
"""Root directory for golden reference GTO population data."""

GEOMETRIES_DIR: Path = Path("./q_block/tests/verification/verification_data/geometries")
"""Directory containing test molecule geometry files (XYZ format)."""


# ======================================================================
# Basis Set Configuration
# ======================================================================

BASIS_FILES: List[str] = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311ppGss.gbs",
]
"""Available GTO basis set filenames in BASIS_ROOT."""

STO_FILES: List[str] = [
    "STO-3G.gbs",
    "STO-6G.gbs",
]
"""Available STO basis set filenames in STO_ROOT."""

# Pre-loaded GTO basis set instances
BASIS_3_21G: Pople = Pople(filepath=str(BASIS_ROOT / "3-21G.gbs"))
"""3-21G split-valence basis set."""

BASIS_6_31G: Pople = Pople(filepath=str(BASIS_ROOT / "6-31G.gbs"))
"""6-31G split-valence basis set."""

BASIS_6_311G: Pople = Pople(filepath=str(BASIS_ROOT / "6-311G.gbs"))
"""6-311G triple-zeta valence basis set."""

BASIS_6_311PP_GSS: Pople = Pople(filepath=str(BASIS_ROOT / "6-311ppGss.gbs"))
"""6-311ppG** basis set with diffuse and polarization functions."""

# Pre-loaded STO basis set instances
BASIS_STO_3G: Pople = Pople(filepath=str(STO_ROOT / "STO-3G.gbs"))
"""STO-3G minimal basis set."""

BASIS_STO_6G: Pople = Pople(filepath=str(STO_ROOT / "STO-6G.gbs"))
"""STO-6G minimal basis set with 6 Gaussians per STO."""

DEFAULT_BASIS: Pople = BASIS_3_21G
"""Default basis set for tests (3-21G)."""


# ======================================================================
# Common Coordinates
# ======================================================================

ORIGIN: CartesianCoordinates = CartesianCoordinates(0.0, 0.0, 0.0)
"""Origin coordinate (0, 0, 0)."""


# ======================================================================
# Test Parameter Values
# ======================================================================

CHARGE_VALUES: List[int] = list(range(-4, 9))
"""Standard charge values for parametrized tests: -4 to +8."""


def charge_ids(charges: List[int] = CHARGE_VALUES) -> List[str]:
    """Generate pytest IDs for charge values.

    :param charges: List of charge values.
    :type charges: List[int]
    :returns: List of formatted ID strings.
    :rtype: List[str]
    """
    return [f"q={q:+d}" if q != 0 else "q=0" for q in charges]
