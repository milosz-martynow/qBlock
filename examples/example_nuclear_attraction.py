"""
Example: Nuclear Attraction Matrix Computation for Water Molecule
=================================================================

This script demonstrates how to:
    1. Build a water molecule with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Extract nuclear positions and charges
    4. Compute the nuclear attraction matrix V
    5. Compute the core Hamiltonian H = T + V
"""

import numpy as np

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.integrals import KineticEnergy, NuclearAttraction, Overlap

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
# 3. EXTRACT NUCLEAR INFORMATION
# ══════════════════════════════════════════════════════════════════════════════

# Build list of (atomic_number, (x, y, z)) for each nucleus
nuclei = [
    (atom.atomic_number, (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z))
    for atom in water.atoms
]

print("=== Nuclear Positions and Charges ===")
for i, (Z, pos) in enumerate(nuclei):
    print(f"  Nucleus {i+1}: Z={Z}, position=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) Bohr")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE NUCLEAR ATTRACTION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

V = NuclearAttraction(
    cgtos=water.contracted_gaussian_type_orbitals,
    nuclei=nuclei,
)

print(f"=== Nuclear Attraction Matrix V ===")
print(f"Shape: {V.matrix.shape}")
print(f"Number of basis functions: {V.n_basis}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VERIFY MATRIX PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

# Symmetry: V_ij = V_ji
is_symmetric = np.allclose(V.matrix, V.matrix.T)
print(f"Symmetric: {is_symmetric}")

# Diagonal elements: nuclear attraction is always negative
diagonal = np.diag(v=V.matrix)
print(f"Diagonal (all negative): {all(d < 0 for d in diagonal)}")
print(f"Diagonal values: {diagonal.round(decimals=4)}")

# Negative semi-definite: all eigenvalues <= 0
eigenvalues = np.linalg.eigvalsh(a=V.matrix)
is_negative_semidefinite = all(ev <= 1e-10 for ev in eigenvalues)
print(f"Negative semi-definite: {is_negative_semidefinite}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMPUTE CORE HAMILTONIAN H = T + V
# ══════════════════════════════════════════════════════════════════════════════

T = KineticEnergy(cgtos=water.contracted_gaussian_type_orbitals)
S = Overlap(cgtos=water.contracted_gaussian_type_orbitals)

H = T.matrix + V.matrix

print("=== Core Hamiltonian H = T + V ===")
print(f"Shape: {H.shape}")
print(f"Symmetric: {np.allclose(H, H.T)}")
print(f"H diagonal values: {np.diag(H).round(decimals=4)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. DISPLAY MATRICES
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=4, suppress=True, linewidth=120)

print("=== Full Nuclear Attraction Matrix V ===")
print(V.matrix)
print()

print("=== Full Core Hamiltonian Matrix H ===")
print(H)
