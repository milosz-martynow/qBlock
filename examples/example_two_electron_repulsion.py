"""
Example: Two-Electron Repulsion Integrals (ERI) for H2 and Water Molecules
==========================================================================

This script demonstrates how to:
    1. Build molecules with different basis sets
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Compute the two-electron repulsion integral (ERI) tensor
    4. Inspect tensor properties (8-fold symmetry)
    5. Understand the role of ERIs in Hartree-Fock
"""

import numpy as np

from q_block.environment.io.basis_set import Pople
from q_block.environment.io.input_data import InputData
from q_block.environment.logs import setup_logging
from q_block.models.integrals import Overlap, TwoElectronRepulsion
from q_block.models.molecule import Molecule

logger = setup_logging(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════

basis_sto3g = Pople(
    filepath="q_block/environment/constants/numerical/basis_set/pople/STO-3G.gbs"
)
basis_3_21G = Pople(
    filepath="q_block/environment/constants/numerical/basis_set/pople/3-21G.gbs"
)
basis_6_31G = Pople(
    filepath="q_block/environment/constants/numerical/basis_set/pople/6-31G.gbs"
)

logger.info("Loaded basis sets: STO-3G, 3-21G and 6-31G\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD H2 MOLECULE (small system for detailed inspection)
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
# 3. BUILD WATER MOLECULE (for comparison with other integral examples)
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
# 4. COMPUTE TWO-ELECTRON REPULSION INTEGRALS (ERI) TENSOR FOR H2
# ══════════════════════════════════════════════════════════════════════════════

ERI_h2 = TwoElectronRepulsion(cgtos=h2.contracted_gaussian_type_orbitals)

logger.info(f"=== H2: Two-Electron Repulsion Integral Tensor ===")
logger.info(f"Shape: {ERI_h2.tensor.shape}")
logger.info(f"Number of basis functions: {ERI_h2.n_basis}")
logger.info(f"Total number of integrals: {ERI_h2.n_basis**4}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPUTE TWO-ELECTRON REPULSION INTEGRALS (ERI) TENSOR FOR WATER
# ══════════════════════════════════════════════════════════════════════════════

ERI_water = TwoElectronRepulsion(cgtos=water.contracted_gaussian_type_orbitals)

logger.info(f"=== Water: Two-Electron Repulsion Integral Tensor ===")
logger.info(f"Shape: {ERI_water.tensor.shape}")
logger.info(f"Number of basis functions: {ERI_water.n_basis}")
logger.info(f"Total number of integrals: {ERI_water.n_basis**4}")
logger.info(f"Unique integrals (8-fold symmetry): ~{ERI_water.n_basis**4 // 8}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. VERIFY 8-FOLD PERMUTATIONAL SYMMETRY (H2)
# ══════════════════════════════════════════════════════════════════════════════

# The ERI tensor has 8-fold symmetry:
# (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)

logger.info("=== H2: Verifying 8-Fold Permutational Symmetry ===")

# Test with specific indices
mu, nu, lam, sig = 0, 1, 0, 1
eri_original = ERI_h2.tensor[mu, nu, lam, sig]

symmetries = [
    ("(μν|λσ)", ERI_h2.tensor[mu, nu, lam, sig]),
    ("(νμ|λσ)", ERI_h2.tensor[nu, mu, lam, sig]),
    ("(μν|σλ)", ERI_h2.tensor[mu, nu, sig, lam]),
    ("(νμ|σλ)", ERI_h2.tensor[nu, mu, sig, lam]),
    ("(λσ|μν)", ERI_h2.tensor[lam, sig, mu, nu]),
    ("(σλ|μν)", ERI_h2.tensor[sig, lam, mu, nu]),
    ("(λσ|νμ)", ERI_h2.tensor[lam, sig, nu, mu]),
    ("(σλ|νμ)", ERI_h2.tensor[sig, lam, nu, mu]),
]

logger.info(f"Testing indices (μ={mu}, ν={nu}, λ={lam}, σ={sig}):")
for name, value in symmetries:
    logger.info(f"  {name} = {value:.6f}")

all_equal = all(np.isclose(val, eri_original) for _, val in symmetries)
logger.info(f"All 8 permutations equal: {all_equal}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PHYSICAL INTERPRETATION OF ERI ELEMENTS (H2)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== H2: Physical Interpretation ===")

# Coulomb integral: (μμ|νν) - electron in orbital μ repels electron in orbital ν
# Exchange integral: (μν|μν) - exchange interaction between orbitals μ and ν

for i in range(ERI_h2.n_basis):
    for j in range(ERI_h2.n_basis):
        coulomb = ERI_h2.tensor[i, i, j, j]
        exchange = ERI_h2.tensor[i, j, i, j]
        logger.info(f"  Coulomb  J[{i},{j}] = ({i}{i}|{j}{j}) = {coulomb:.6f}")
        logger.info(f"  Exchange K[{i},{j}] = ({i}{j}|{i}{j}) = {exchange:.6f}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 8. RELATIONSHIP TO HARTREE-FOCK
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=== Role in Hartree-Fock Theory ===")
logger.info("The two-electron integrals are used to construct:")
logger.info("  - Coulomb matrix J: J_μν = Σ_λσ P_λσ (μν|λσ)")
logger.info("  - Exchange matrix K: K_μν = Σ_λσ P_λσ (μλ|νσ)")
logger.info("  - Fock matrix F = H + J - K (restricted HF)")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 9. COMPARE WITH ONE-ELECTRON INTEGRALS
# ══════════════════════════════════════════════════════════════════════════════

S_h2 = Overlap(cgtos=h2.contracted_gaussian_type_orbitals)
S_water = Overlap(cgtos=water.contracted_gaussian_type_orbitals)

logger.info("=== Comparison with One-Electron Integrals ===")
logger.info(
    f"H2:    Overlap S shape: {S_h2.matrix.shape} "
    f"(2D), ERI shape: {ERI_h2.tensor.shape} (4D)"
)
logger.info(
    f"Water: Overlap S shape: {S_water.matrix.shape} "
    f"(2D), ERI shape: {ERI_water.tensor.shape} (4D)"
)
logger.info(f"S is 2-center integral: ⟨μ|ν⟩")
logger.info(f"ERI is 4-center integral: (μν|λσ)\n")


# ══════════════════════════════════════════════════════════════════════════════
# 10. DISPLAY H2 ERI TENSOR
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=6, suppress=True, linewidth=120)

logger.info("=== H2: Full ERI Tensor (in chemist's notation) ===")
logger.info("Format: (μν|λσ)")
for mu in range(ERI_h2.n_basis):
    for nu in range(ERI_h2.n_basis):
        for lam in range(ERI_h2.n_basis):
            for sig in range(ERI_h2.n_basis):
                val = ERI_h2.tensor[mu, nu, lam, sig]
                if abs(val) > 1e-10:
                    logger.info(f"  ({mu}{nu}|{lam}{sig}) = {val:.6f}")
