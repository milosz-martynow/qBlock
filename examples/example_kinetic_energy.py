"""
Example: Kinetic Energy Matrix Computation for Water Molecule
=============================================================

This script demonstrates how to:
    1. Build a water molecule with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Compute the kinetic energy matrix T
    4. Inspect matrix properties (symmetry, positive semi-definiteness)
"""

import numpy as np

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.integrals import KineticEnergy, Overlap

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
# 3. COMPUTE KINETIC ENERGY MATRIX
# ══════════════════════════════════════════════════════════════════════════════

T = KineticEnergy(cgtos=water.contracted_gaussian_type_orbitals)

print(f"=== Kinetic Energy Matrix T ===")
print(f"Shape: {T.matrix.shape}")
print(f"Number of basis functions: {T.n_basis}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFY MATRIX PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

# Symmetry: T_ij = T_ji
is_symmetric = np.allclose(T.matrix, T.matrix.T)
print(f"Symmetric: {is_symmetric}")

# Diagonal elements: kinetic energy is always positive
diagonal = np.diag(v=T.matrix)
print(f"Diagonal (all positive): {all(d > 0 for d in diagonal)}")
print(f"Diagonal values: {diagonal.round(decimals=4)}")

# Positive semi-definite: all eigenvalues >= 0
eigenvalues = np.linalg.eigvalsh(a=T.matrix)
is_positive_semidefinite = all(ev >= -1e-10 for ev in eigenvalues)
print(f"Positive semi-definite: {is_positive_semidefinite}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARE WITH OVERLAP MATRIX
# ══════════════════════════════════════════════════════════════════════════════

S = Overlap(cgtos=water.contracted_gaussian_type_orbitals)

print("=== Comparison with Overlap Matrix S ===")
print(f"Same shape: {T.matrix.shape == S.matrix.shape}")
print(f"T diagonal / S diagonal (should be positive):")
print(f"  {(np.diag(T.matrix) / np.diag(S.matrix)).round(decimals=4)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DISPLAY KINETIC ENERGY MATRIX
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("=== Full Kinetic Energy Matrix T ===")
print(T.matrix)
