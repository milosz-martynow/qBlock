"""
Example: SCF Hartree-Fock – Full Pipeline
==========================================

Demonstrates the complete qBlock workflow from raw input to saved output,
driven by a ``.qblock.config.example`` configuration file:

    Configuration  →  Input Data  →  Model  →  Solver (SCF)  →  Output Data

Three Hartree-Fock variants are shown:

    1. RHF  – Restricted Closed-Shell on H₂ (singlet, 2 electrons)
    2. UHF  – Unrestricted on H₂⁺ (doublet, 1 electron)
    3. ROHF – Restricted Open-Shell on H₂⁺ (doublet, 1 electron)

The base configuration (basis set, geometry, SCF parameters) is loaded
from ``examples/.qblock.config.example``.  The UHF and ROHF runs reuse
the same geometry but override charge and multiplicity programmatically.
Results are saved as JSON and formatted text files in an
``output_example_scf_hartree_fock/`` folder next to this script.
"""

import importlib.resources
from pathlib import Path
from typing import List, Tuple

from q_block.compute.environment.configuration import Configuration
from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.io.output_data import OutputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.models.initialization import RHF, ROHF, UHF
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.models.molecule import Molecule
from q_block.compute.solvers.wavefunction.hartree_fock import (
    RestrictedHartreeFock,
    RestrictedOpenShellHartreeFock,
    UnrestrictedHartreeFock,
)

logger = setup_logging(__name__)

BASIS_DIR: Path = (
    Path(str(importlib.resources.files("q_block.compute.environment.constants.numerical")))
    / "basis_set"
    / "pople"
)
SCRIPT_DIR: Path = Path(str(importlib.resources.files("q_block.learn.examples")))
SCRIPT_NAME: str = "example_scf_hartree_fock"

OUTPUT_DIR = SCRIPT_DIR / f"output_{SCRIPT_NAME}"
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
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CONFIG_PATH = SCRIPT_DIR / ".qblock.config.example"
config = Configuration.from_file(CONFIG_PATH)
logger.info(f"Loaded configuration: {config}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT DATA (from configuration)
# ══════════════════════════════════════════════════════════════════════════════

basis = Pople(
    filepath=str(BASIS_DIR / f"{config.get_basis_set()}.gbs")
)
logger.info(f"Loaded basis set: {config.get_basis_set()}\n")

# Build atom_data from config geometry, attaching the loaded basis set
atom_data_h2 = [
    [row[0], row[1], row[2], row[3], basis]
    for row in config.get_geometry()
]

# ── H₂ (neutral, singlet) from config ───────────────────────────────────────
inp_h2 = InputData()
inp_h2.from_script(atom_data=atom_data_h2)

# ── H₂⁺ (cation, doublet) – same geometry, override charge ─────────────────
atom_data_h2_cat = [
    [row[0], row[1], row[2], row[3], basis, -1]
    for row in config.get_geometry()[:1]
] + [
    [row[0], row[1], row[2], row[3], basis]
    for row in config.get_geometry()[1:]
]

inp_h2_cat = InputData()
inp_h2_cat.from_script(atom_data=atom_data_h2_cat)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL – Initialization Contexts
# ══════════════════════════════════════════════════════════════════════════════

# ── RHF context for H₂ (charge/multiplicity from config) ────────────────────
mol_h2 = Molecule(
    input_data=inp_h2,
    multiplicity=config.get_multiplicity(),
)
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
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
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
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
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
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
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
    rhf,
    method="RHF",
    context=ctx_rhf,
    basis_set=config.get_basis_set(),
)
if config.get_save_json():
    out_rhf.to_json(OUTPUT_DIR / "output_rhf_h2.json")
    logger.info(f"  RHF  → {OUTPUT_DIR / 'output_rhf_h2.json'}")
if config.get_save_text():
    out_rhf.save_text(OUTPUT_DIR / "output_rhf_h2.txt")
    logger.info(f"  RHF  → {OUTPUT_DIR / 'output_rhf_h2.txt'}")

# ── UHF output ───────────────────────────────────────────────────────────────
out_uhf = OutputData.from_scf(
    uhf,
    method="UHF",
    context=ctx_uhf,
    basis_set=config.get_basis_set(),
)
if config.get_save_json():
    out_uhf.to_json(OUTPUT_DIR / "output_uhf_h2_cation.json")
    logger.info(f"  UHF  → {OUTPUT_DIR / 'output_uhf_h2_cation.json'}")
if config.get_save_text():
    out_uhf.save_text(OUTPUT_DIR / "output_uhf_h2_cation.txt")
    logger.info(f"  UHF  → {OUTPUT_DIR / 'output_uhf_h2_cation.txt'}")

# ── ROHF output ──────────────────────────────────────────────────────────────
out_rohf = OutputData.from_scf(
    rohf,
    method="ROHF",
    context=ctx_rohf,
    basis_set=config.get_basis_set(),
)
if config.get_save_json():
    out_rohf.to_json(OUTPUT_DIR / "output_rohf_h2_cation.json")
    logger.info(f"  ROHF → {OUTPUT_DIR / 'output_rohf_h2_cation.json'}")
if config.get_save_text():
    out_rohf.save_text(OUTPUT_DIR / "output_rohf_h2_cation.txt")
    logger.info(f"  ROHF → {OUTPUT_DIR / 'output_rohf_h2_cation.txt'}")

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
