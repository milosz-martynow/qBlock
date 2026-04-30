"""
Example: SCF Kohn-Sham DFT – Full Pipeline
===========================================

Demonstrates the qBlock workflow for Density Functional Theory (DFT)
calculations:

    Configuration  →  Input Data  →  Model  →  Solver (KS-SCF)  →  Output

Three exchange-correlation functionals are shown on H₂ (singlet):

    1. RKS-SVWN  – Local Density Approximation (LDA)
    2. RKS-PBE   – Generalised Gradient Approximation (GGA)
    3. RKS-B3LYP – Hybrid functional (GGA + exact exchange)

The base configuration (basis set, geometry, SCF parameters) is loaded
from ``examples/.qblock.config.example``.  Results are saved as JSON and
formatted text files in an ``output_example_scf_kohn_sham/`` folder next
to this script.
"""

from pathlib import Path
from typing import List, Tuple

from q_block.compute.environment.configuration import Configuration
from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.io.output_data import OutputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.models.initialization import RKS
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.models.molecule import Molecule
from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
)

logger = setup_logging(__name__)

try:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    SCRIPT_DIR: Path = Path(__file__).resolve().parent
    SCRIPT_NAME: str = Path(__file__).stem
except NameError:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == "examples":
        SCRIPT_DIR = PROJECT_ROOT
        PROJECT_ROOT = PROJECT_ROOT.parent.parent.parent.parent
    elif (PROJECT_ROOT / "q_block" / "q_block" / "q_block" / "learn" / "examples").exists():
        SCRIPT_DIR = PROJECT_ROOT / "q_block" / "q_block" / "q_block" / "learn" / "examples"
    else:
        raise RuntimeError(
            "Could not determine project root. "
            "Please run from project root or examples directory."
        )
    SCRIPT_NAME = "example_scf_kohn_sham"

BASIS_DIR: Path = (
    PROJECT_ROOT
    / "q_block"
    / "compute"
    / "environment"
    / "constants"
    / "numerical"
    / "basis_set"
    / "pople"
)

OUTPUT_DIR = SCRIPT_DIR / f"output_{SCRIPT_NAME}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

CONFIG_PATH = SCRIPT_DIR / ".qblock.config.example"
config = Configuration.from_file(CONFIG_PATH)
logger.info(f"Loaded configuration: {config}")


# ══════════════════════════════════════════════════════════════════
# HELPER: extract nuclei list from a Molecule
# ══════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════
# 1. INPUT DATA (from configuration)
# ══════════════════════════════════════════════════════════════════

basis = Pople(
    filepath=str(BASIS_DIR / f"{config.get_basis_set()}.gbs")
)
logger.info(f"Loaded basis set: {config.get_basis_set()}\n")

atom_data_h2 = [
    [row[0], row[1], row[2], row[3], basis]
    for row in config.get_geometry()
]

inp_h2 = InputData()
inp_h2.from_script(atom_data=atom_data_h2)


# ══════════════════════════════════════════════════════════════════
# 2. MODEL – Initialization Context
# ══════════════════════════════════════════════════════════════════

mol_h2 = Molecule(
    input_data=inp_h2,
    multiplicity=config.get_multiplicity(),
)
ctx_rks = RKS(molecule=mol_h2)
logger.info(f"RKS context: {ctx_rks}")

nuclei_h2 = extract_nuclei(ctx_rks.molecule)
e_nuc_h2 = NuclearRepulsionEnergy(ctx_rks.molecule).energy


# ══════════════════════════════════════════════════════════════════
# 3. SOLVER – SCF Kohn-Sham
# ══════════════════════════════════════════════════════════════════

# ── RKS-SVWN (LDA) ──────────────────────────────────────────────
logger.info("=" * 70)
logger.info("RKS-SVWN – Local Density Approximation (H₂, singlet)")
logger.info("=" * 70)

rks_svwn = RestrictedKohnSham(
    cgtos=ctx_rks.cgto,
    nuclei=nuclei_h2,
    e_nuclear=e_nuc_h2,
    functional=SVWN(),
    n_electrons=ctx_rks.n_electrons,
    n_radial=50,
    n_angular=14,
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
).run()

logger.info(f"  Converged          : {rks_svwn.converged}")
logger.info(f"  Iterations         : {rks_svwn.n_iterations}")
logger.info(
    f"  Total energy       : {rks_svwn.e_total:.10f} Hartree\n"
)

# ── RKS-PBE (GGA) ───────────────────────────────────────────────
logger.info("=" * 70)
logger.info(
    "RKS-PBE – Generalised Gradient Approximation (H₂, singlet)"
)
logger.info("=" * 70)

rks_pbe = RestrictedKohnSham(
    cgtos=ctx_rks.cgto,
    nuclei=nuclei_h2,
    e_nuclear=e_nuc_h2,
    functional=PBE(),
    n_electrons=ctx_rks.n_electrons,
    n_radial=50,
    n_angular=14,
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
).run()

logger.info(f"  Converged          : {rks_pbe.converged}")
logger.info(f"  Iterations         : {rks_pbe.n_iterations}")
logger.info(
    f"  Total energy       : {rks_pbe.e_total:.10f} Hartree\n"
)

# ── RKS-B3LYP (hybrid) ─────────────────────────────────────────
logger.info("=" * 70)
logger.info("RKS-B3LYP – Hybrid functional (H₂, singlet)")
logger.info("=" * 70)

rks_b3lyp = RestrictedKohnSham(
    cgtos=ctx_rks.cgto,
    nuclei=nuclei_h2,
    e_nuclear=e_nuc_h2,
    functional=B3LYP(),
    n_electrons=ctx_rks.n_electrons,
    n_radial=50,
    n_angular=14,
    max_iterations=config.get_max_scf_iterations(),
    convergence_threshold=config.get_scf_convergence_threshold(),
    diis_start=config.get_diis_start(),
    diis_max_vectors=config.get_diis_max_vectors(),
    calculation_error_metric=config.get_error_metric(),
).run()

logger.info(f"  Converged          : {rks_b3lyp.converged}")
logger.info(f"  Iterations         : {rks_b3lyp.n_iterations}")
logger.info(
    f"  Total energy       : {rks_b3lyp.e_total:.10f} Hartree\n"
)


# ══════════════════════════════════════════════════════════════════
# 4. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Summary: H₂ total energies (Hartree)")
logger.info("=" * 70)
logger.info(f"  SVWN  (LDA)    : {rks_svwn.e_total:.10f}")
logger.info(f"  PBE   (GGA)    : {rks_pbe.e_total:.10f}")
logger.info(f"  B3LYP (hybrid) : {rks_b3lyp.e_total:.10f}\n")


# ══════════════════════════════════════════════════════════════════
# 5. OUTPUT DATA – Save Results
# ══════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Saving output data")
logger.info("=" * 70)

for label, rks in [
    ("svwn", rks_svwn),
    ("pbe", rks_pbe),
    ("b3lyp", rks_b3lyp),
]:
    out = OutputData.from_scf(
        rks,
        method=f"RKS-{label.upper()}",
        context=ctx_rks,
        basis_set=config.get_basis_set(),
    )
    if config.get_save_json():
        path = OUTPUT_DIR / f"output_rks_{label}_h2.json"
        out.to_json(path)
        logger.info(f"  RKS-{label.upper()} → {path}")
    if config.get_save_text():
        path = OUTPUT_DIR / f"output_rks_{label}_h2.txt"
        out.save_text(path)
        logger.info(f"  RKS-{label.upper()} → {path}")
