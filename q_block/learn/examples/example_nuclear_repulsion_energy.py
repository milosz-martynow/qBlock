"""
Example: Nuclear Repulsion Energy Computation for H2 and Water Molecules
========================================================================

This script demonstrates how to:
    1. Build molecules with different basis sets
    2. Convert coordinates to Bohr (atomic units)
    3. Compute the nuclear repulsion energy E_nuc
    4. Verify that E_nuc is independent of basis set choice

Nuclear repulsion energy is a classical Coulomb sum between nuclei:

    E_nuc = Σ_{A<B} (Z_A * Z_B) / R_AB

It does NOT involve basis functions — only geometry and atomic numbers.
"""

import importlib.resources
import math
from pathlib import Path

from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.models.initialization import NuclearRepulsionEnergy
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
h2.to_bohr()  # Convert to atomic units (REQUIRED before E_nuc computation)

logger.info(f"H2 molecule: {h2.n_electrons} electrons")
logger.info(f"Number of atoms: {len(h2.atoms)}\n")


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
water.to_bohr()  # Convert to atomic units (REQUIRED before E_nuc computation)

logger.info(f"Water molecule: {water.n_electrons} electrons")
logger.info(f"Number of atoms: {len(water.atoms)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE NUCLEAR REPULSION ENERGIES
# ══════════════════════════════════════════════════════════════════════════════

E_nuc_h2 = NuclearRepulsionEnergy(h2)
E_nuc_water = NuclearRepulsionEnergy(water)

logger.info(f"=== H2: Nuclear Repulsion Energy ===")
logger.info(f"Number of atoms: {E_nuc_h2.n_atoms}")
logger.info(f"E_nuc = {E_nuc_h2.energy:.6f} Hartree\n")

logger.info(f"=== Water: Nuclear Repulsion Energy ===")
logger.info(f"Number of atoms: {E_nuc_water.n_atoms}")
logger.info(f"E_nuc = {E_nuc_water.energy:.6f} Hartree\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DISPLAY NUCLEAR POSITIONS (H2)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== H2: Nuclear Positions and Charges ===")
for i, (Z, pos) in enumerate(E_nuc_h2.nuclei):
    logger.info(
        f"  Nucleus {i+1}: Z={Z}, "
        f"position=({pos[0]:.4f}, "
        f"{pos[1]:.4f}, {pos[2]:.4f}) Bohr"
    )
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DISPLAY NUCLEAR POSITIONS (Water)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Water: Nuclear Positions and Charges ===")
for i, (Z, pos) in enumerate(E_nuc_water.nuclei):
    symbol = "O" if Z == 8 else "H"
    logger.info(
        f"  {symbol} (Z={Z}): ({pos[0]:.4f}, " f"{pos[1]:.4f}, {pos[2]:.4f}) Bohr"
    )
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 7. VERIFY MANUAL CALCULATION (H2)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== H2: Manual Verification ===")

# For H2: E_nuc = Z_A * Z_B / R_AB = 1 * 1 / R_AB
R_AB = math.sqrt(
    (E_nuc_h2.nuclei[0][1][0] - E_nuc_h2.nuclei[1][1][0]) ** 2
    + (E_nuc_h2.nuclei[0][1][1] - E_nuc_h2.nuclei[1][1][1]) ** 2
    + (E_nuc_h2.nuclei[0][1][2] - E_nuc_h2.nuclei[1][1][2]) ** 2
)
E_manual = 1.0 / R_AB

logger.info(f"  Bond length R_AB = {R_AB:.6f} Bohr")
logger.info(f"  E_nuc (computed) = {E_nuc_h2.energy:.6f} Hartree")
logger.info(f"  E_nuc (manual)   = {E_manual:.6f} Hartree")
logger.info(f"  Match: {abs(E_nuc_h2.energy - E_manual) < 1e-10}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PAIRWISE CONTRIBUTIONS (Water)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Water: Pairwise Contributions ===")
symbols = ["O", "H1", "H2"]
total = 0.0

for a in range(E_nuc_water.n_atoms):
    Z_a, R_a = E_nuc_water.nuclei[a]
    for b in range(a + 1, E_nuc_water.n_atoms):
        Z_b, R_b = E_nuc_water.nuclei[b]
        E_pair = NuclearRepulsionEnergy.pairwise_energy(Z_a, R_a, Z_b, R_b)
        total += E_pair
        logger.info(f"  {symbols[a]}-{symbols[b]}: {E_pair:.6f} Hartree")

logger.info(f"  ─────────────────────")
logger.info(f"  Total:   {total:.6f} Hartree")
logger.info(f"  E_nuc:   {E_nuc_water.energy:.6f} Hartree\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. VERIFY BASIS SET INDEPENDENCE
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Basis Set Independence Test ===")
logger.info("E_nuc should be IDENTICAL regardless of basis set choice.\n")

# Build same water geometry with different basis sets
water_sto3g_input = InputData()
water_sto3g_input.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_sto3g],
        ["H", 0.0000, 0.7572, -0.4692, basis_sto3g],
        ["H", 0.0000, -0.7572, -0.4692, basis_sto3g],
    ]
)
water_sto3g = Molecule(input_data=water_sto3g_input)
water_sto3g.to_bohr()

water_321g_input = InputData()
water_321g_input.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_3_21G],
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G],
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G],
    ]
)
water_321g = Molecule(input_data=water_321g_input)
water_321g.to_bohr()

water_631g_input = InputData()
water_631g_input.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G],
        ["H", 0.0000, 0.7572, -0.4692, basis_6_31G],
        ["H", 0.0000, -0.7572, -0.4692, basis_6_31G],
    ]
)
water_631g = Molecule(input_data=water_631g_input)
water_631g.to_bohr()

E_sto3g = NuclearRepulsionEnergy(water_sto3g)
E_321g = NuclearRepulsionEnergy(water_321g)
E_631g = NuclearRepulsionEnergy(water_631g)

logger.info(f"  Water (STO-3G): E_nuc = {E_sto3g.energy:.6f} Hartree")
logger.info(f"  Water (3-21G):  E_nuc = {E_321g.energy:.6f} Hartree")
logger.info(f"  Water (6-31G):  E_nuc = {E_631g.energy:.6f} Hartree")
all_identical = (
    abs(E_sto3g.energy - E_321g.energy) < 1e-10
    and abs(E_321g.energy - E_631g.energy) < 1e-10
)
logger.info(f"\n  All identical: {all_identical}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10. OBJECT FEATURES
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Object Features ===")
logger.info(f"  repr(E_nuc_h2):    {repr(E_nuc_h2)}")
logger.info(f"  repr(E_nuc_water): {repr(E_nuc_water)}")
logger.info(f"  float(E_nuc_h2):   {float(E_nuc_h2):.6f}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 11. KEY POINTS
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 60)
logger.info("KEY POINTS:")
logger.info("=" * 60)
logger.info(
    """
1. Nuclear repulsion energy is CLASSICAL (no quantum mechanics)
2. It depends ONLY on:
   - Nuclear charges (atomic numbers)
   - Nuclear positions
3. It does NOT depend on:
   - Basis set (STO-3G, 3-21G, 6-31G all give same result!)
   - HF method (RHF/UHF/ROHF)
   - Electron count
4. Coordinates MUST be in Bohr before computation
5. E_nuc is always POSITIVE (repulsive interaction)
6. Total HF energy: E_total = E_electronic + E_nuc
"""
)
