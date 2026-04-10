"""
Example: Nuclear Attraction Matrix Computation for H2 and Water Molecules
=========================================================================

This script demonstrates how to:
    1. Build molecules with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Extract nuclear positions and charges
    4. Compute the nuclear attraction matrix V
    5. Compute the core Hamiltonian H = T + V
"""

import logging

import numpy as np

from q_block.environment.io.basis_set import Pople
from q_block.environment.io.input_data import InputData
from q_block.models.molecule import Molecule
from q_block.models.integrals import KineticEnergy, NuclearAttraction

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════

basis_sto3g = Pople(filepath="q_block/environment/constants/numerical/basis_set/pople/STO-3G.gbs")
basis_3_21G = Pople(filepath="q_block/environment/constants/numerical/basis_set/pople/3-21G.gbs")
basis_6_31G = Pople(filepath="q_block/environment/constants/numerical/basis_set/pople/6-31G.gbs")

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
# 4. EXTRACT NUCLEAR INFORMATION
# ══════════════════════════════════════════════════════════════════════════════

# Build list of (atomic_number, (x, y, z)) for each nucleus
h2_nuclei = [
    (
        atom.atomic_number,
        (
            atom.coordinates.x,
            atom.coordinates.y,
            atom.coordinates.z,
        ),
    )
    for atom in h2.atoms
]

water_nuclei = [
    (
        atom.atomic_number,
        (
            atom.coordinates.x,
            atom.coordinates.y,
            atom.coordinates.z,
        ),
    )
    for atom in water.atoms
]

logger.info("=== H2: Nuclear Positions and Charges ===")
for i, (Z, pos) in enumerate(h2_nuclei):
    logger.info(
        f"  Nucleus {i+1}: Z={Z}, "
        f"position=({pos[0]:.4f}, "
        f"{pos[1]:.4f}, {pos[2]:.4f}) Bohr"
    )
logger.info("")

logger.info("=== Water: Nuclear Positions and Charges ===")
for i, (Z, pos) in enumerate(water_nuclei):
    logger.info(
        f"  Nucleus {i+1}: Z={Z}, "
        f"position=({pos[0]:.4f}, "
        f"{pos[1]:.4f}, {pos[2]:.4f}) Bohr"
    )
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE NUCLEAR ATTRACTION MATRICES
# ══════════════════════════════════════════════════════════════════════════════

V_h2 = NuclearAttraction(
    cgtos=h2.contracted_gaussian_type_orbitals,
    nuclei=h2_nuclei,
)

V_water = NuclearAttraction(
    cgtos=water.contracted_gaussian_type_orbitals,
    nuclei=water_nuclei,
)

logger.info(f"=== H2: Nuclear Attraction Matrix V ===")
logger.info(f"Shape: {V_h2.matrix.shape}")
logger.info(f"Number of basis functions: {V_h2.n_basis}\n")

logger.info(f"=== Water: Nuclear Attraction Matrix V ===")
logger.info(f"Shape: {V_water.matrix.shape}")
logger.info(f"Number of basis functions: {V_water.n_basis}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. VERIFY MATRIX PROPERTIES (Water)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Water: Matrix Properties ===")

# Symmetry: V_ij = V_ji
is_symmetric = np.allclose(V_water.matrix, V_water.matrix.T)
logger.info(f"Symmetric: {is_symmetric}")

# Diagonal elements: nuclear attraction is always negative
diagonal = np.diag(v=V_water.matrix)
logger.info(f"Diagonal (all negative): {all(d < 0 for d in diagonal)}")
logger.info(f"Diagonal values: {diagonal.round(decimals=4)}")

# Negative semi-definite: all eigenvalues <= 0
eigenvalues = np.linalg.eigvalsh(a=V_water.matrix)
is_negative_semidefinite = all(ev <= 1e-10 for ev in eigenvalues)
logger.info(f"Negative semi-definite: {is_negative_semidefinite}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPUTE CORE HAMILTONIAN H = T + V
# ══════════════════════════════════════════════════════════════════════════════

T_h2 = KineticEnergy(cgtos=h2.contracted_gaussian_type_orbitals)
T_water = KineticEnergy(cgtos=water.contracted_gaussian_type_orbitals)

H_h2 = T_h2.matrix + V_h2.matrix
H_water = T_water.matrix + V_water.matrix

logger.info("=== H2: Core Hamiltonian H = T + V ===")
logger.info(f"Shape: {H_h2.shape}")
logger.info(f"Symmetric: {np.allclose(H_h2, H_h2.T)}")
logger.info(f"H diagonal values: {np.diag(H_h2).round(decimals=4)}\n")

logger.info("=== Water: Core Hamiltonian H = T + V ===")
logger.info(f"Shape: {H_water.shape}")
logger.info(f"Symmetric: {np.allclose(H_water, H_water.T)}")
logger.info(f"H diagonal values: {np.diag(H_water).round(decimals=4)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DISPLAY MATRICES
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=4, suppress=True, linewidth=120)

logger.info("=== H2: Full Nuclear Attraction Matrix V ===")
logger.info(f"\n{V_h2.matrix}")
logger.info("")

logger.info("=== H2: Full Core Hamiltonian Matrix H ===")
logger.info(f"\n{H_h2}")
logger.info("")

logger.info("=== Water: Full Nuclear Attraction Matrix V ===")
logger.info(f"\n{V_water.matrix}")
logger.info("")

logger.info("=== Water: Full Core Hamiltonian Matrix H ===")
logger.info(f"\n{H_water}")
