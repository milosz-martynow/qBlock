"""
Example: Creating and Inspecting a Molecule
============================================

This script demonstrates how to create a Molecule from a Python structure
using the InputData helper class.

Steps:
    1. Create InputData with atom definitions
    2. Build a Molecule from InputData
    3. Inspect molecule properties
"""

import logging

from q_block.io.input_data import InputData
from q_block.models.molecule import Molecule

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEFINE ATOMS USING InputData
# ══════════════════════════════════════════════════════════════════════════════
# InputData.from_script() accepts a list of atom definitions.
# Each entry is: [symbol, x, y, z]
# Coordinates are in Ångströms.

water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O", 0.000000, 0.000000, 0.000000],  # Oxygen at origin
        ["H", 0.758602, 0.000000, 0.504284],  # Hydrogen 1
        ["H", -0.758602, 0.000000, 0.504284],  # Hydrogen 2
    ],
    atom_prefix="W",  # Prefix for atom IDs (W1, W2, W3)
)

logger.info("=== InputData atoms DataFrame ===")
logger.info(f"\n{water_input.atoms}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MOLECULE FROM InputData
# ══════════════════════════════════════════════════════════════════════════════
# The Molecule class wraps InputData and provides molecular properties.
# It extracts Atom instances and computes aggregate quantities.

water = Molecule(input_data=water_input)

logger.info("=== Molecule object ===")
logger.info(water)
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INSPECT MOLECULE PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Molecule properties ===")
logger.info(f"  Number of atoms:     {len(water.atoms)}")
logger.info(f"  Total atomic number: {water.total_atomic_number}")
logger.info(f"  Total electrons:     {water.n_electrons}")
logger.info(f"  Charge:              {water.charge}")
logger.info(f"  Multiplicity:        {water.multiplicity}")
logger.info("")

# Access individual atoms
logger.info("=== Individual atoms ===")
for i, atom in enumerate(water.atoms):
    logger.info(f"  Atom {i}: {atom}")
