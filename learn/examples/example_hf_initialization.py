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

from compute.environment.io.basis_set import Pople
from compute.environment.io.input_data import InputData
from compute.environment.logs import setup_logging
from compute.models.initialization import RHF, ROHF, UHF
from compute.models.molecule import Molecule

logger = setup_logging(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SETS
# ══════════════════════════════════════════════════════════════════════════════
# We use different basis sets for oxygen and hydrogen to demonstrate
# that mixed basis sets are supported.

basis_3_21G = Pople(
    filepath="compute/environment/constants/numerical/basis_set/pople/3-21G.gbs"
)
basis_6_31G = Pople(
    filepath="compute/environment/constants/numerical/basis_set/pople/6-31G.gbs"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. RHF - RESTRICTED CLOSED-SHELL (NEUTRAL WATER, SINGLET)
# ══════════════════════════════════════════════════════════════════════════════
# RHF is appropriate for closed-shell molecules (all electrons paired).
# Neutral water has 10 electrons -> 5 doubly-occupied orbitals.

logger.info("=" * 70)
logger.info("RHF - Restricted Closed-Shell (neutral H2O, singlet)")
logger.info("=" * 70)

# Create InputData for neutral water
# Entry format: [symbol, x, y, z, basis_set, charge]
# charge=0 (default) means neutral atom
inp_rhf = InputData()
inp_rhf.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G, 0],  # Oxygen with 6-31G
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G, 0],  # H with 3-21G
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G, 0],  # H with 3-21G
    ]
)

# Build molecule with singlet multiplicity (default)
water_rhf = Molecule(input_data=inp_rhf, multiplicity=1)

# Initialize RHF calculation
rhf = RHF(molecule=water_rhf)

logger.info(rhf)
logger.info("")
logger.info(f"  Total electrons          : {rhf.n_electrons}")
logger.info(f"  Occupied orbitals (n_occ): {rhf.n_occ}")
logger.info(f"  Alpha electrons          : {rhf.n_alpha}")
logger.info(f"  Beta electrons           : {rhf.n_beta}")
logger.info(f"  Basis functions          : {rhf.n_basis}")
logger.info(f"  HF method                : {rhf.hf_method}")
logger.info(f"  Charge                   : {rhf.charge}")
logger.info(f"  Multiplicity             : {rhf.multiplicity}")

# Coordinates are converted to Bohr during initialization
first_atom = rhf.molecule.atoms[0]
logger.info(f"  First atom coords (Bohr) : {first_atom.coordinates}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. UHF - UNRESTRICTED (WATER CATION H2O+, DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# UHF allows different spatial orbitals for alpha and beta electrons.
# Water cation has 9 electrons -> doublet (one unpaired electron).

logger.info("=" * 70)
logger.info("UHF - Unrestricted (H2O+ cation, doublet)")
logger.info("=" * 70)

# Remove one electron by setting oxygen charge to -1
# (negative charge = fewer electrons than protons)
inp_uhf = InputData()
inp_uhf.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G, -1],  # O with -1 charge
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G, 0],
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G, 0],
    ]
)

# Build molecule with doublet multiplicity (2S+1 = 2 -> S = 1/2)
water_uhf = Molecule(input_data=inp_uhf, multiplicity=2)

# Initialize UHF calculation
uhf = UHF(molecule=water_uhf)

logger.info(uhf)
logger.info("")
logger.info(f"  Total electrons          : {uhf.n_electrons}")
logger.info(f"  Alpha electrons          : {uhf.n_alpha}")
logger.info(f"  Beta electrons           : {uhf.n_beta}")
logger.info(f"  Basis functions          : {uhf.n_basis}")
logger.info(f"  HF method                : {uhf.hf_method}")
logger.info(f"  Charge                   : {uhf.charge}")
logger.info(f"  Multiplicity             : {uhf.multiplicity}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROHF - RESTRICTED OPEN-SHELL (WATER CATION H2O+, DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# ROHF uses the same spatial orbitals for paired electrons but allows
# open-shell configuration. Gives spin-pure wavefunctions.

logger.info("=" * 70)
logger.info("ROHF - Restricted Open-Shell (H2O+ cation, doublet)")
logger.info("=" * 70)

# Same cation as UHF
inp_rohf = InputData()
inp_rohf.from_script(
    atom_data=[
        ["O", 0.0000, 0.0000, 0.1173, basis_6_31G, -1],
        ["H", 0.0000, 0.7572, -0.4692, basis_3_21G, 0],
        ["H", 0.0000, -0.7572, -0.4692, basis_3_21G, 0],
    ]
)

water_rohf = Molecule(input_data=inp_rohf, multiplicity=2)

# Initialize ROHF calculation
rohf = ROHF(molecule=water_rohf)

logger.info(rohf)
logger.info("")
logger.info(f"  Total electrons          : {rohf.n_electrons}")
logger.info(f"  Alpha electrons          : {rohf.n_alpha}")
logger.info(f"  Beta electrons           : {rohf.n_beta}")
logger.info(f"  Closed-shell orbitals    : {rohf.n_closed}")
logger.info(f"  Open-shell orbitals      : {rohf.n_open}")
logger.info(f"  Basis functions          : {rohf.n_basis}")
logger.info(f"  HF method                : {rohf.hf_method}")
logger.info(f"  Charge                   : {rohf.charge}")
logger.info(f"  Multiplicity             : {rohf.multiplicity}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Comparison of HF methods for water")
logger.info("=" * 70)

header = f"{'':>20}  {'RHF':>6}  {'UHF':>6}  {'ROHF':>6}"
logger.info(header)
logger.info("-" * len(header))
logger.info(
    f"{'Charge':>20}  {rhf.charge:>6}" f"  {uhf.charge:>6}  {rohf.charge:>6}"
)
logger.info(
    f"{'Multiplicity':>20}  {rhf.multiplicity:>6}"
    f"  {uhf.multiplicity:>6}  {rohf.multiplicity:>6}"
)
logger.info(
    f"{'Electrons':>20}  {rhf.n_electrons:>6}"
    f"  {uhf.n_electrons:>6}  {rohf.n_electrons:>6}"
)
logger.info(
    f"{'Alpha':>20}  {rhf.n_alpha:>6}" f"  {uhf.n_alpha:>6}  {rohf.n_alpha:>6}"
)
logger.info(f"{'Beta':>20}  {rhf.n_beta:>6}" f"  {uhf.n_beta:>6}  {rohf.n_beta:>6}")
logger.info(
    f"{'Basis functions':>20}  {rhf.n_basis:>6}"
    f"  {uhf.n_basis:>6}  {rohf.n_basis:>6}"
)
logger.info(
    f"{'HF method':>20}  {rhf.hf_method:>6}"
    f"  {uhf.hf_method:>6}  {rohf.hf_method:>6}"
)
