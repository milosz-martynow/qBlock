"""
Example: Overlap Matrix Computation for Water Molecule
======================================================

This script demonstrates how to:
    1. Build a water molecule with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Compute the overlap matrix S
    4. Inspect matrix properties (symmetry, normalization)
"""

import numpy as np

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.integrals import Overlap

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════

basis_3_21G = Pople(filepath="data/basis_set/gto_gaussian_format/3-21G.gbs")
basis_6_31G = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")

print("Loaded basis sets: 3-21G and 6-31G\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD WATER MOLECULE
# ══════════════════════════════════════════════════════════════════════════════

water_input = InputData()
water_input.from_script(atom_data=[
    ["O", 0.0000, 0.0000, 0.1173, basis_6_31G],
    ["H", 0.0000, 0.7572, -0.4692, basis_3_21G],
    ["H", 0.0000, -0.7572, -0.4692, basis_3_21G],
])

water = Molecule(input_data=water_input)
water.to_bohr()  # Convert to atomic units (required for integrals)
water.make_contracted_gaussian_type_orbital()

print(f"Water molecule: {water.n_electrons} electrons")
print(f"Number of CGTOs: {len(water.contracted_gaussian_type_orbitals)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE OVERLAP MATRIX
# ══════════════════════════════════════════════════════════════════════════════

S = Overlap(cgtos=water.contracted_gaussian_type_orbitals)

print(f"=== Overlap Matrix S ===")
print(f"Shape: {S.matrix.shape}")
print(f"Number of basis functions: {S.n_basis}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFY MATRIX PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

# Symmetry: S_ij = S_ji
is_symmetric = np.allclose(S.matrix, S.matrix.T)
print(f"Symmetric: {is_symmetric}")

# Normalization: diagonal elements should be ~1.0
diagonal = np.diag(v=S.matrix)
print(f"Diagonal (normalization): {diagonal.round(decimals=4)}")

# Positive definite: all eigenvalues > 0
eigenvalues = np.linalg.eigvalsh(a=S.matrix)
is_positive_definite = all(ev > 0 for ev in eigenvalues)
print(f"Positive definite: {is_positive_definite}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DISPLAY OVERLAP MATRIX
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("=== Full Overlap Matrix ===")
print(S.matrix)
