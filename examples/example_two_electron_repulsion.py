"""
Example: Two-Electron Repulsion Integrals (ERI) for H2 Molecule
===============================================================

This script demonstrates how to:
    1. Build a H2 molecule with a basis set
    2. Generate contracted Gaussian-type orbitals (CGTOs)
    3. Compute the two-electron repulsion integral (ERI) tensor
    4. Inspect tensor properties (8-fold symmetry)
    5. Understand the role of ERIs in Hartree-Fock
"""

import numpy as np

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.integrals import TwoElectronRepulsion, Overlap

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SET
# ══════════════════════════════════════════════════════════════════════════════

basis_sto3g = Pople(filepath="data/basis_set/sto_gaussian_format/STO-3G.gbs")

print("Loaded basis set: STO-3G\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD H2 MOLECULE
# ══════════════════════════════════════════════════════════════════════════════

# Use H2 molecule (small basis for fast computation)
h2_input = InputData()
h2_input.from_script(atom_data=[
    ["H", 0.0000, 0.0000, 0.0000, basis_sto3g],
    ["H", 0.0000, 0.0000, 0.7414, basis_sto3g],  # Bond length ~0.74 Angstrom
])

h2 = Molecule(input_data=h2_input)
h2.to_bohr()  # Convert to atomic units (required for integrals)
h2.make_contracted_gaussian_type_orbital()

print(f"H2 molecule: {h2.n_electrons} electrons")
print(f"Number of CGTOs: {len(h2.contracted_gaussian_type_orbitals)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE TWO-ELECTRON REPULSION INTEGRALS (ERI) TENSOR
# ══════════════════════════════════════════════════════════════════════════════

ERI = TwoElectronRepulsion(cgtos=h2.contracted_gaussian_type_orbitals)

print(f"=== Two-Electron Repulsion Integral Tensor ===")
print(f"Shape: {ERI.tensor.shape}")
print(f"Number of basis functions: {ERI.n_basis}")
print(f"Total number of integrals: {ERI.n_basis**4}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFY 8-FOLD PERMUTATIONAL SYMMETRY
# ══════════════════════════════════════════════════════════════════════════════

# The ERI tensor has 8-fold symmetry:
# (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)

print("=== Verifying 8-Fold Permutational Symmetry ===")

# Test with specific indices
mu, nu, lam, sig = 0, 1, 0, 1
eri_original = ERI.tensor[mu, nu, lam, sig]

symmetries = [
    ("(μν|λσ)", ERI.tensor[mu, nu, lam, sig]),
    ("(νμ|λσ)", ERI.tensor[nu, mu, lam, sig]),
    ("(μν|σλ)", ERI.tensor[mu, nu, sig, lam]),
    ("(νμ|σλ)", ERI.tensor[nu, mu, sig, lam]),
    ("(λσ|μν)", ERI.tensor[lam, sig, mu, nu]),
    ("(σλ|μν)", ERI.tensor[sig, lam, mu, nu]),
    ("(λσ|νμ)", ERI.tensor[lam, sig, nu, mu]),
    ("(σλ|νμ)", ERI.tensor[sig, lam, nu, mu]),
]

print(f"Testing indices (μ={mu}, ν={nu}, λ={lam}, σ={sig}):")
for name, value in symmetries:
    print(f"  {name} = {value:.6f}")

all_equal = all(np.isclose(val, eri_original) for _, val in symmetries)
print(f"All 8 permutations equal: {all_equal}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PHYSICAL INTERPRETATION OF ERI ELEMENTS
# ══════════════════════════════════════════════════════════════════════════════

print("=== Physical Interpretation ===")

# Coulomb integral: (μμ|νν) - electron in orbital μ repels electron in orbital ν
# Exchange integral: (μν|μν) - exchange interaction between orbitals μ and ν

for i in range(ERI.n_basis):
    for j in range(ERI.n_basis):
        coulomb = ERI.tensor[i, i, j, j]
        exchange = ERI.tensor[i, j, i, j]
        print(f"  Coulomb  J[{i},{j}] = ({i}{i}|{j}{j}) = {coulomb:.6f}")
        print(f"  Exchange K[{i},{j}] = ({i}{j}|{i}{j}) = {exchange:.6f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 6. RELATIONSHIP TO HARTREE-FOCK
# ══════════════════════════════════════════════════════════════════════════════

print("=== Role in Hartree-Fock Theory ===")
print("The two-electron integrals are used to construct:")
print("  - Coulomb matrix J: J_μν = Σ_λσ P_λσ (μν|λσ)")
print("  - Exchange matrix K: K_μν = Σ_λσ P_λσ (μλ|νσ)")
print("  - Fock matrix F = H + J - K (restricted HF)")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPARE WITH ONE-ELECTRON INTEGRALS
# ══════════════════════════════════════════════════════════════════════════════

S = Overlap(cgtos=h2.contracted_gaussian_type_orbitals)

print("=== Comparison with One-Electron Integrals ===")
print(f"Overlap matrix S shape: {S.matrix.shape} (2D)")
print(f"ERI tensor shape: {ERI.tensor.shape} (4D)")
print(f"S is 2-center integral: ⟨μ|ν⟩")
print(f"ERI is 4-center integral: (μν|λσ)\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DISPLAY ERI TENSOR
# ══════════════════════════════════════════════════════════════════════════════

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("=== Full ERI Tensor (in chemist's notation) ===")
print("Format: (μν|λσ)")
for mu in range(ERI.n_basis):
    for nu in range(ERI.n_basis):
        for lam in range(ERI.n_basis):
            for sig in range(ERI.n_basis):
                val = ERI.tensor[mu, nu, lam, sig]
                if abs(val) > 1e-10:
                    print(f"  ({mu}{nu}|{lam}{sig}) = {val:.6f}")
