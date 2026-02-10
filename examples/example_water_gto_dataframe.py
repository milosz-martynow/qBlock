"""Example: water molecule and spin-orbital DataFrame

This example shows how to construct a `Molecule` with different
Pople basis sets on hydrogen and oxygen atoms, and then export the
occupied spin-orbitals into a single `pandas.DataFrame` using
`Molecule.to_dataframe`.
"""

import pandas as pd

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule

# ── 1. Create Pople basis set objects ────────────────────────────────
basis_3_21G = Pople(filepath="data/basis_set/gto_gaussian_format/3-21G.gbs")
basis_6_31G = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")

# ── 2. Define atoms: [symbol, x, y, z, basis_set] ───────────────────
water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O",  0.0000,  0.0000,  0.1173, basis_6_31G],
        ["H",  0.0000,  0.7572, -0.4692, basis_3_21G],
        ["H",  0.0000, -0.7572, -0.4692, basis_3_21G],
    ],
)

# ── 3. Build the Molecule (GTO population automatic) ─────────────────
water = Molecule(input_data=water_input)

# ── 4. Export occupied spin-orbital data to a multi-index DataFrame ──
spinorb_df = water.to_dataframe()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(spinorb_df)
