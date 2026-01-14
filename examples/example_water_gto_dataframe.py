"""Example: water molecule and spin-orbital DataFrame

This example shows how to construct a `Molecule` from an XYZ geometry,
attach different Pople basis sets to hydrogen and oxygen atoms,
and then export the occupied spin-orbitals into a single
`pandas.DataFrame` using `Molecule.to_dataframe`.
"""

from pathlib import Path

import pandas as pd

from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule
from q_block.io.basis_set_pople import parse_gaussian_basis


# Work from the project root directory
project_root = Path.cwd()

xyz_path = project_root / "tests" / "verification_data" / "geometries" / "water.xyz"
basis_3_21G_path = project_root / "data" / "basis_set" / "gto_gaussian_format" / "3-21G.gbs"
basis_6_31G_path = project_root / "data" / "basis_set" / "gto_gaussian_format" / "6-31G.gbs"

# 1. Load geometry into InputData
input_data = InputData()
input_data.from_xyz_file(xyz_path=xyz_path)

# 2. Load Pople basis sets
basis_3_21G = parse_gaussian_basis(filepath=str(basis_3_21G_path))
basis_6_31G = parse_gaussian_basis(filepath=str(basis_6_31G_path))

# 3. Attach basis sets to atoms: 3-21G for H, 6-31G for O
for row in input_data.atoms.itertuples():
    atom = row.atom
    symbol = row.symbol

    if symbol == "H":
        atom.basis_set = basis_3_21G["H"]
    elif symbol == "O":
        atom.basis_set = basis_6_31G["O"]

# 4. Build the Molecule; spin-orbitals are populated with GTO data
water = Molecule(input_data=input_data)

# 5. Export occupied spin-orbital data to a multi-index DataFrame
spinorb_df = water.to_dataframe()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(spinorb_df)
