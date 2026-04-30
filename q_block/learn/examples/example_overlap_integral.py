"""
Example: Overlap Matrix Computation for H2 and Water Molecules
==============================================================

This script demonstrates how to:
    1. Build molecules with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Compute the overlap matrix S
    4. Inspect matrix properties (symmetry, normalization)
"""

import importlib.resources
from pathlib import Path

import numpy as np

from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.models.integrals import Overlap
from q_block.compute.models.molecule import Molecule

logger = setup_logging(__name__)

BASIS_DIR: Path = (
    Path(str(importlib.resources.files("q_block.compute.environment.constants.numerical")))
    / "basis_set"
    / "pople"
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════

basis_sto3g = Pople(filepath=str(BASIS_DIR / "STO-3G.gbs"))
basis_3_21G = Pople(filepath=str(BASIS_DIR / "3-21G.gbs"))
basis_6_31G = Pople(filepath=str(BASIS_DIR / "6-31G.gbs"))

logger.info("Loaded basis sets: STO-3G, 3-21G and 6-31G\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD H2 MOLECULE
# ══════════════════════════════════════════════════════════════════════════════

h2_input = InputData()
h2_input.from_script(
    atom_data=[
        ["H", 0.0000, 0.0000, 0.0000, basis_sto3g],
        ["H", 0.0000, 0.0000, 0.7414, basis_sto3g],  # Bond length ~0.74 Angstrom
    ]
)

h2 = Molecule(input_data=h2_input)
h2.to_bohr()  # Convert to atomic units (required for integrals)
h2.make_contracted_gaussian_type_orbital()

logger.info(f"H2 molecule: {h2.n_electrons} electrons")
logger.info(f"Number of CGTOs: {len(h2.contracted_gaussian_type_orbitals)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD WATER MOLECULE
# ══════════════════════════════════════════════════════════════════════════════

water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G],
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G],
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G],
    ]
)

water = Molecule(input_data=water_input)
water.to_bohr()  # Convert to atomic units (required for integrals)
water.make_contracted_gaussian_type_orbital()

logger.info(f"Water molecule: {water.n_electrons} electrons")
logger.info(f"Number of CGTOs: {len(water.contracted_gaussian_type_orbitals)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE OVERLAP MATRICES
# ══════════════════════════════════════════════════════════════════════════════

S_h2 = Overlap(cgtos=h2.contracted_gaussian_type_orbitals)
S_water = Overlap(cgtos=water.contracted_gaussian_type_orbitals)

logger.info(f"=== H2: Overlap Matrix S ===")
logger.info(f"Shape: {S_h2.matrix.shape}")
logger.info(f"Number of basis functions: {S_h2.n_basis}\n")

logger.info(f"=== Water: Overlap Matrix S ===")
logger.info(f"Shape: {S_water.matrix.shape}")
logger.info(f"Number of basis functions: {S_water.n_basis}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VERIFY MATRIX PROPERTIES (Water)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Water: Matrix Properties ===")

# Symmetry: S_ij = S_ji
is_symmetric = np.allclose(S_water.matrix, S_water.matrix.T)
logger.info(f"Symmetric: {is_symmetric}")

# Normalization: diagonal elements should be ~1.0
diagonal = np.diag(v=S_water.matrix)
logger.info(f"Diagonal (normalization): {diagonal.round(decimals=4)}")

# Positive definite: all eigenvalues > 0
eigenvalues = np.linalg.eigvalsh(a=S_water.matrix)
is_positive_definite = all(ev > 0 for ev in eigenvalues)
logger.info(f"Positive definite: {is_positive_definite}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DISPLAY OVERLAP MATRICES
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=4, suppress=True, linewidth=120)

logger.info("=== H2: Full Overlap Matrix ===")
logger.info(f"\n{S_h2.matrix}")
logger.info("")

logger.info("=== Water: Full Overlap Matrix ===")
logger.info(f"\n{S_water.matrix}")
