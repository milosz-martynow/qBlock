"""
Example: Water Molecule Spin-Orbital DataFrame Export
======================================================

This script demonstrates how to:
    1. Build a water molecule with mixed basis sets
    2. Export occupied spin-orbital data to a pandas DataFrame
    3. Inspect the electronic structure

The to_dataframe() method creates a MultiIndex DataFrame containing
all occupied spin-orbitals with their quantum numbers and GTO data.
"""

from pathlib import Path

import pandas as pd

from compute.environment.io.basis_set import Pople
from compute.environment.io.input_data import InputData
from compute.environment.logs import setup_logging
from compute.models.molecule import Molecule

logger = setup_logging(__name__)

try:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "examples":
        PROJECT_ROOT = PROJECT_ROOT.parent.parent
    elif (PROJECT_ROOT / "learn" / "examples").exists():
        pass
    else:
        raise RuntimeError(
            "Could not determine project root. "
            "Please run from project root or examples directory."
        )

BASIS_DIR: Path = (
    PROJECT_ROOT
    / "compute"
    / "environment"
    / "constants"
    / "numerical"
    / "basis_set"
    / "pople"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════
# Different basis sets for different atoms (mixed basis).

basis_3_21G = Pople(filepath=str(BASIS_DIR / "3-21G.gbs"))
basis_6_31G = Pople(filepath=str(BASIS_DIR / "6-31G.gbs"))

logger.info("Loaded basis sets: 3-21G and 6-31G\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE WATER MOLECULE
# ══════════════════════════════════════════════════════════════════════════════
# Atom entry format: [symbol, x, y, z, basis_set]
# Coordinates in Ångströms.

water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G],  # Oxygen with 6-31G
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G],  # H1 with 3-21G
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G],  # H2 with 3-21G
    ]
)

logger.info("=== Atom input data ===")
logger.info(f"\n{water_input.atoms[['symbol', 'x', 'y', 'z']]}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MOLECULE
# ══════════════════════════════════════════════════════════════════════════════
# The Molecule constructor automatically populates GTO data for atoms
# that have an attached basis set.

water = Molecule(input_data=water_input)

logger.info("=== Molecule summary ===")
logger.info(f"  Atoms:           {len(water.atoms)}")
logger.info(f"  Total electrons: {water.n_electrons}")
logger.info(f"  Charge:          {water.charge}")
logger.info(f"  Multiplicity:    {water.multiplicity}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT TO DataFrame
# ══════════════════════════════════════════════════════════════════════════════
# to_dataframe() returns a MultiIndex DataFrame with:
#   - Index levels: atom_id, n, l, m, spin
#   - Columns: occupied, exponents, coefficients, etc.
#
# This gives a complete view of the electronic structure.

spinorb_df = water.to_dataframe()

# Configure pandas display for wide output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)

logger.info("=== Spin-orbital DataFrame ===")
logger.info(f"\n{spinorb_df}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ACCESS SPECIFIC DATA
# ══════════════════════════════════════════════════════════════════════════════
# The MultiIndex allows easy querying of specific orbitals.

logger.info("=== DataFrame shape ===")
logger.info(f"  Rows: {len(spinorb_df)}")
logger.info(f"  Columns: {list(spinorb_df.columns.get_level_values(0).unique())}")
logger.info("")

logger.info("=== Index levels ===")
logger.info(f"  Levels: {spinorb_df.index.names}")
