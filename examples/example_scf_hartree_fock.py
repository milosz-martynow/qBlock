"""
Example: SCF Hartree-Fock – Full Pipeline
==========================================

Demonstrates the complete qBlock workflow from raw input to saved output:

    Input Data  →  Model (initialization)  →  Solver (SCF)  →  Output Data

Three Hartree-Fock variants are shown:

    1. RHF  – Restricted Closed-Shell on H₂ (singlet, 2 electrons)
    2. UHF  – Unrestricted on H₂⁺ (doublet, 1 electron)
    3. ROHF – Restricted Open-Shell on H₂⁺ (doublet, 1 electron)

All calculations use the 3-21G basis set.  Results are saved as JSON and
formatted text files in an ``output_example_scf_hartree_fock/`` folder
next to this script.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from q_block.environment.io.basis_set import Pople
from q_block.environment.io.input_data import InputData
from q_block.environment.io.output_data import OutputData
from q_block.models.molecule import Molecule
from q_block.models.initialization import RHF, ROHF, UHF
from q_block.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.solvers.wavefunction.hartree_fock import (
    RestrictedHartreeFock,
    RestrictedOpenShellHartreeFock,
    UnrestrictedHartreeFock,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory = subfolder next to this script, named after the script
OUTPUT_DIR = Path(__file__).resolve().parent / f"output_{Path(__file__).stem}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: extract nuclei list from a Molecule
# ══════════════════════════════════════════════════════════════════════════════


def extract_nuclei(
    mol: Molecule,
) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Return a list of (Z, (x, y, z)) tuples for each nucleus.

    :param mol: Molecule in Bohr units.
    :type mol: Molecule
    :returns: List of (atomic_number, (x, y, z)) pairs.
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return [
        (
            atom.atomic_number,
            (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z),
        )
        for atom in mol.atoms
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT DATA
# ══════════════════════════════════════════════════════════════════════════════

basis_3_21G = Pople(
    filepath="q_block/environment/constants/numerical/basis_set/pople/3-21G.gbs"
)
logger.info("Loaded basis set: 3-21G\n")

# ── H₂ (neutral, singlet) ───────────────────────────────────────────────────
inp_h2 = InputData()
inp_h2.from_script(
    atom_data=[
        ["H", 0.0, 0.0, 0.0, basis_3_21G],
        ["H", 0.0, 0.0, 0.74, basis_3_21G],
    ]
)

# ── H₂⁺ (cation, doublet) ──────────────────────────────────────────────────
inp_h2_cat = InputData()
inp_h2_cat.from_script(
    atom_data=[
        ["H", 0.0, 0.0, 0.0, basis_3_21G, -1],
        ["H", 0.0, 0.0, 0.74, basis_3_21G],
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL – Initialization Contexts
# ══════════════════════════════════════════════════════════════════════════════

# ── RHF context for H₂ ──────────────────────────────────────────────────────
mol_h2 = Molecule(input_data=inp_h2, multiplicity=1)
ctx_rhf = RHF(molecule=mol_h2)
logger.info(f"RHF context: {ctx_rhf}")

# ── UHF context for H₂⁺ ────────────────────────────────────────────────────
mol_h2_cat_uhf = Molecule(input_data=inp_h2_cat, multiplicity=2)
ctx_uhf = UHF(molecule=mol_h2_cat_uhf)
logger.info(f"UHF context: {ctx_uhf}")

# ── ROHF context for H₂⁺ ───────────────────────────────────────────────────
mol_h2_cat_rohf = Molecule(input_data=inp_h2_cat, multiplicity=2)
ctx_rohf = ROHF(molecule=mol_h2_cat_rohf)
logger.info(f"ROHF context: {ctx_rohf}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SOLVER – SCF Hartree-Fock
# ══════════════════════════════════════════════════════════════════════════════

# ── RHF on H₂ ───────────────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("RHF – Restricted Closed-Shell (H₂, singlet)")
logger.info("=" * 70)

nuclei_h2 = extract_nuclei(ctx_rhf.molecule)
e_nuc_h2 = NuclearRepulsionEnergy(ctx_rhf.molecule).energy

rhf = RestrictedHartreeFock(
    cgtos=ctx_rhf.cgto,
    nuclei=nuclei_h2,
    e_nuclear=e_nuc_h2,
    n_electrons=ctx_rhf.n_electrons,
).run()

logger.info(f"  Converged          : {rhf.converged}")
logger.info(f"  Iterations         : {rhf.n_iterations}")
logger.info(f"  Total energy       : {rhf.e_total:.10f} Hartree\n")

# ── UHF on H₂⁺ ──────────────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("UHF – Unrestricted (H₂⁺ cation, doublet)")
logger.info("=" * 70)

nuclei_cat = extract_nuclei(ctx_uhf.molecule)
e_nuc_cat = NuclearRepulsionEnergy(ctx_uhf.molecule).energy

uhf = UnrestrictedHartreeFock(
    cgtos=ctx_uhf.cgto,
    nuclei=nuclei_cat,
    e_nuclear=e_nuc_cat,
    n_alpha=ctx_uhf.n_alpha,
    n_beta=ctx_uhf.n_beta,
).run()

logger.info(f"  Converged          : {uhf.converged}")
logger.info(f"  Iterations         : {uhf.n_iterations}")
logger.info(f"  Total energy       : {uhf.e_total:.10f} Hartree\n")

# ── ROHF on H₂⁺ ─────────────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("ROHF – Restricted Open-Shell (H₂⁺ cation, doublet)")
logger.info("=" * 70)

rohf = RestrictedOpenShellHartreeFock(
    cgtos=ctx_rohf.cgto,
    nuclei=nuclei_cat,
    e_nuclear=e_nuc_cat,
    n_closed=ctx_rohf.n_closed,
    n_open=ctx_rohf.n_open,
).run()

logger.info(f"  Converged          : {rohf.converged}")
logger.info(f"  Iterations         : {rohf.n_iterations}")
logger.info(f"  Total energy       : {rohf.e_total:.10f} Hartree\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OUTPUT DATA – Save Results
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Saving output data")
logger.info("=" * 70)

# ── RHF output ───────────────────────────────────────────────────────────────
out_rhf = OutputData.from_scf(
    rhf, method="RHF", context=ctx_rhf, basis_set="3-21G",
)
out_rhf.to_json(OUTPUT_DIR / "output_rhf_h2.json")
out_rhf.save_text(OUTPUT_DIR / "output_rhf_h2.txt")
logger.info(f"  RHF  → {OUTPUT_DIR / 'output_rhf_h2.json'}")
logger.info(f"       → {OUTPUT_DIR / 'output_rhf_h2.txt'}")

# ── UHF output ───────────────────────────────────────────────────────────────
out_uhf = OutputData.from_scf(
    uhf, method="UHF", context=ctx_uhf, basis_set="3-21G",
)
out_uhf.to_json(OUTPUT_DIR / "output_uhf_h2_cation.json")
out_uhf.save_text(OUTPUT_DIR / "output_uhf_h2_cation.txt")
logger.info(f"  UHF  → {OUTPUT_DIR / 'output_uhf_h2_cation.json'}")
logger.info(f"       → {OUTPUT_DIR / 'output_uhf_h2_cation.txt'}")

# ── ROHF output ──────────────────────────────────────────────────────────────
out_rohf = OutputData.from_scf(
    rohf, method="ROHF", context=ctx_rohf, basis_set="3-21G",
)
out_rohf.to_json(OUTPUT_DIR / "output_rohf_h2_cation.json")
out_rohf.save_text(OUTPUT_DIR / "output_rohf_h2_cation.txt")
logger.info(f"  ROHF → {OUTPUT_DIR / 'output_rohf_h2_cation.json'}")
logger.info(f"       → {OUTPUT_DIR / 'output_rohf_h2_cation.txt'}")

logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON: UHF vs ROHF FOR H₂⁺
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Comparison: UHF vs ROHF for H₂⁺")
logger.info("=" * 70)

delta = abs(uhf.e_total - rohf.e_total)
logger.info(f"  UHF  total energy  : {uhf.e_total:.10f} Hartree")
logger.info(f"  ROHF total energy  : {rohf.e_total:.10f} Hartree")
logger.info(f"  |UHF - ROHF|       : {delta:.2e} Hartree")
logger.info(f"  Energies agree     : {delta < 1e-10}")
