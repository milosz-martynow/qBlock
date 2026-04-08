"""
Example: Hartree-Fock Quantum Calculation Context for Water Molecule
====================================================================

This script demonstrates how to use RHF, UHF, and ROHF quantum calculation
context classes for Hartree-Fock calculations.

Three scenarios are shown:
    1. RHF - Restricted Closed-Shell (neutral water, singlet)
    2. UHF - Unrestricted (water cation H2O+, doublet)
    3. ROHF - Restricted Open-Shell (water cation H2O+, doublet)

Each HF context:
    - Validates the molecule has basis sets on all atoms
    - Computes electron counts (alpha, beta)
    - Converts coordinates to Bohr
    - Prepares data structures for SCF calculation
"""

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.initialization import RHF, UHF, ROHF

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════
# We use different basis sets for oxygen and hydrogen to demonstrate
# that mixed basis sets are supported.

basis_3_21G = Pople(filepath="data/basis_set/pople/3-21G.gbs")
basis_6_31G = Pople(filepath="data/basis_set/pople/6-31G.gbs")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RHF - RESTRICTED CLOSED-SHELL (NEUTRAL WATER, SINGLET)
# ══════════════════════════════════════════════════════════════════════════════
# RHF is appropriate for closed-shell molecules (all electrons paired).
# Neutral water has 10 electrons -> 5 doubly-occupied orbitals.

print("=" * 70)
print("RHF - Restricted Closed-Shell (neutral H2O, singlet)")
print("=" * 70)

# Create InputData for neutral water
# Entry format: [symbol, x, y, z, basis_set, charge]
# charge=0 (default) means neutral atom
inp_rhf = InputData()
inp_rhf.from_script(atom_data=[
    ["O",  0.0000,  0.0000,  0.1173, basis_6_31G, 0],  # Oxygen with 6-31G
    ["H",  0.0000,  0.7572, -0.4692, basis_3_21G, 0],  # H with 3-21G
    ["H",  0.0000, -0.7572, -0.4692, basis_3_21G, 0],  # H with 3-21G
])

# Build molecule with singlet multiplicity (default)
water_rhf = Molecule(input_data=inp_rhf, multiplicity=1)

# Initialize RHF calculation
rhf = RHF(molecule=water_rhf)

print(rhf)
print()
print(f"  Total electrons          : {rhf.n_electrons}")
print(f"  Occupied orbitals (n_occ): {rhf.n_occ}")
print(f"  Alpha electrons          : {rhf.n_alpha}")
print(f"  Beta electrons           : {rhf.n_beta}")
print(f"  Basis functions          : {rhf.n_basis}")
print(f"  HF method                : {rhf.hf_method}")
print(f"  Charge                   : {rhf.charge}")
print(f"  Multiplicity             : {rhf.multiplicity}")

# Coordinates are converted to Bohr during initialization
first_atom = rhf.molecule.atoms[0]
print(f"  First atom coords (Bohr) : {first_atom.coordinates}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. UHF - UNRESTRICTED (WATER CATION H2O+, DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# UHF allows different spatial orbitals for alpha and beta electrons.
# Water cation has 9 electrons -> doublet (one unpaired electron).

print("=" * 70)
print("UHF - Unrestricted (H2O+ cation, doublet)")
print("=" * 70)

# Remove one electron by setting oxygen charge to -1
# (negative charge = fewer electrons than protons)
inp_uhf = InputData()
inp_uhf.from_script(atom_data=[
    ["O",  0.0000,  0.0000,  0.1173, basis_6_31G, -1],  # O with -1 charge
    ["H",  0.0000,  0.7572, -0.4692, basis_3_21G, 0],
    ["H",  0.0000, -0.7572, -0.4692, basis_3_21G, 0],
])

# Build molecule with doublet multiplicity (2S+1 = 2 -> S = 1/2)
water_uhf = Molecule(input_data=inp_uhf, multiplicity=2)

# Initialize UHF calculation
uhf = UHF(molecule=water_uhf)

print(uhf)
print()
print(f"  Total electrons          : {uhf.n_electrons}")
print(f"  Alpha electrons          : {uhf.n_alpha}")
print(f"  Beta electrons           : {uhf.n_beta}")
print(f"  Basis functions          : {uhf.n_basis}")
print(f"  HF method                : {uhf.hf_method}")
print(f"  Charge                   : {uhf.charge}")
print(f"  Multiplicity             : {uhf.multiplicity}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROHF - RESTRICTED OPEN-SHELL (WATER CATION H2O+, DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# ROHF uses the same spatial orbitals for paired electrons but allows
# open-shell configuration. Gives spin-pure wavefunctions.

print("=" * 70)
print("ROHF - Restricted Open-Shell (H2O+ cation, doublet)")
print("=" * 70)

# Same cation as UHF
inp_rohf = InputData()
inp_rohf.from_script(atom_data=[
    ["O",  0.0000,  0.0000,  0.1173, basis_6_31G, -1],
    ["H",  0.0000,  0.7572, -0.4692, basis_3_21G, 0],
    ["H",  0.0000, -0.7572, -0.4692, basis_3_21G, 0],
])

water_rohf = Molecule(input_data=inp_rohf, multiplicity=2)

# Initialize ROHF calculation
rohf = ROHF(molecule=water_rohf)

print(rohf)
print()
print(f"  Total electrons          : {rohf.n_electrons}")
print(f"  Alpha electrons          : {rohf.n_alpha}")
print(f"  Beta electrons           : {rohf.n_beta}")
print(f"  Closed-shell orbitals    : {rohf.n_closed}")
print(f"  Open-shell orbitals      : {rohf.n_open}")
print(f"  Basis functions          : {rohf.n_basis}")
print(f"  HF method                : {rohf.hf_method}")
print(f"  Charge                   : {rohf.charge}")
print(f"  Multiplicity             : {rohf.multiplicity}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Comparison of HF methods for water")
print("=" * 70)

header = f"{'':>20}  {'RHF':>6}  {'UHF':>6}  {'ROHF':>6}"
print(header)
print("-" * len(header))
print(f"{'Charge':>20}  {rhf.charge:>6}  {uhf.charge:>6}  {rohf.charge:>6}")
print(f"{'Multiplicity':>20}  {rhf.multiplicity:>6}  {uhf.multiplicity:>6}  {rohf.multiplicity:>6}")
print(f"{'Electrons':>20}  {rhf.n_electrons:>6}  {uhf.n_electrons:>6}  {rohf.n_electrons:>6}")
print(f"{'Alpha':>20}  {rhf.n_alpha:>6}  {uhf.n_alpha:>6}  {rohf.n_alpha:>6}")
print(f"{'Beta':>20}  {rhf.n_beta:>6}  {uhf.n_beta:>6}  {rohf.n_beta:>6}")
print(f"{'Basis functions':>20}  {rhf.n_basis:>6}  {uhf.n_basis:>6}  {rohf.n_basis:>6}")
print(f"{'HF method':>20}  {rhf.hf_method:>6}  {uhf.hf_method:>6}  {rohf.hf_method:>6}")
