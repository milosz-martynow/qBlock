"""
Example: SCF Hartree-Fock Calculations (RHF, UHF, ROHF)
========================================================

This script demonstrates how to run Self-Consistent Field (SCF)
calculations for all three Hartree-Fock variants:

    1. RHF  – Restricted Closed-Shell on H2 (singlet, 2 electrons)
    2. UHF  – Unrestricted on H2⁺ (doublet, 1 electron)
    3. ROHF – Restricted Open-Shell on H2⁺ (doublet, 1 electron)

All calculations use the 3-21G basis set.  The UHF and ROHF results
for H2⁺ are compared to verify they converge to the same energy.
"""

import logging
from typing import List, Tuple

from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.solvers.wavefunction.hartree_fock import (
    RestrictedHartreeFock,
    RestrictedOpenShellHartreeFock,
    UnrestrictedHartreeFock,
)
from q_block.models.molecule import Molecule
from q_block.models.initialization import RHF, ROHF, UHF
from q_block.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SET
# ══════════════════════════════════════════════════════════════════════════════

basis_3_21G = Pople(filepath="q_block/constants/numerical/basis_set/pople/3-21G.gbs")
logger.info("Loaded basis set: 3-21G\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: build molecule and extract SCF inputs
# ══════════════════════════════════════════════════════════════════════════════


def prepare_molecule(
    atom_data: List[list],
    multiplicity: int = 1,
) -> Tuple[
    List,
    List[Tuple[int, Tuple[float, float, float]]],
    float,
]:
    """Build a Molecule and return (cgtos, nuclei, e_nuclear).

    :param atom_data: List of [symbol, x, y, z, basis_set, charge]
        entries.
    :type atom_data: List[list]
    :param multiplicity: Spin multiplicity (2S+1).
    :type multiplicity: int
    :returns: (cgtos, nuclei, e_nuclear) ready for HF
        constructors.
    :rtype: Tuple[List, List[Tuple[int, Tuple[float, float, float]]], float]
    """
    inp = InputData()
    inp.from_script(atom_data=atom_data)
    mol = Molecule(input_data=inp, multiplicity=multiplicity)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    cgtos = mol.contracted_gaussian_type_orbitals
    nuclei = [
        (
            atom.atomic_number,
            (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z),
        )
        for atom in mol.atoms
    ]
    e_nuclear = NuclearRepulsionEnergy(mol).energy
    return cgtos, nuclei, e_nuclear


# ══════════════════════════════════════════════════════════════════════════════
# 2. RHF – RESTRICTED CLOSED-SHELL ON H2 (SINGLET)
# ══════════════════════════════════════════════════════════════════════════════
# H2 has 2 electrons -> 1 doubly-occupied orbital -> singlet.

logger.info("=" * 70)
logger.info("RHF - Restricted Closed-Shell (H2, singlet)")
logger.info("=" * 70)

cgtos, nuclei, e_nuclear = prepare_molecule(
    atom_data=[
        ["H", 0.0, 0.0, 0.0, basis_3_21G],
        ["H", 0.0, 0.0, 0.74, basis_3_21G],  # bond length ~0.74 Å
    ],
    multiplicity=1,
)

rhf = RestrictedHartreeFock(
    cgtos=cgtos,
    nuclei=nuclei,
    e_nuclear=e_nuclear,
    n_electrons=2,
).run()

logger.info(rhf)
logger.info("")
logger.info(f"  Converged          : {rhf.converged}")
logger.info(f"  Iterations         : {rhf.n_iterations}")
logger.info(f"  Electronic energy  : {rhf.e_electronic:.10f} Hartree")
logger.info(f"  Nuclear repulsion  : {e_nuclear:.10f} Hartree")
logger.info(f"  Total energy       : {rhf.e_total:.10f} Hartree")
logger.info(f"  Stored matrices    : {sorted(rhf.matrices)}")
logger.info(f"  Orbital energies   : {rhf.matrices['epsilon']}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. UHF – UNRESTRICTED ON H2⁺ (DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# H2⁺ has 1 electron -> doublet (one unpaired alpha electron).
# Charge is set on one hydrogen atom (charge=-1 removes one electron).

logger.info("=" * 70)
logger.info("UHF – Unrestricted (H2⁺ cation, doublet)")
logger.info("=" * 70)

cgtos_cat, nuclei_cat, e_nuclear_cat = prepare_molecule(
    atom_data=[
        ["H", 0.0, 0.0, 0.0, basis_3_21G, -1],  # -1 charge removes 1 electron
        ["H", 0.0, 0.0, 0.74, basis_3_21G],
    ],
    multiplicity=2,
)

uhf = UnrestrictedHartreeFock(
    cgtos=cgtos_cat,
    nuclei=nuclei_cat,
    e_nuclear=e_nuclear_cat,
    n_alpha=1,
    n_beta=0,
).run()

logger.info(uhf)
logger.info("")
logger.info(f"  Converged          : {uhf.converged}")
logger.info(f"  Iterations         : {uhf.n_iterations}")
logger.info(f"  Electronic energy  : {uhf.e_electronic:.10f} Hartree")
logger.info(f"  Nuclear repulsion  : {e_nuclear_cat:.10f} Hartree")
logger.info(f"  Total energy       : {uhf.e_total:.10f} Hartree")
logger.info(f"  Stored matrices    : {sorted(uhf.matrices)}")
logger.info(f"  Alpha orb. energies: {uhf.matrices['epsilon_alpha']}")
logger.info(f"  Beta  orb. energies: {uhf.matrices['epsilon_beta']}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROHF – RESTRICTED OPEN-SHELL ON H2⁺ (DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# Same H2⁺ system, but with shared spatial orbitals and the Roothaan
# effective Fock matrix.  n_closed=0 (no doubly-occupied), n_open=1.

logger.info("=" * 70)
logger.info("ROHF – Restricted Open-Shell (H2⁺ cation, doublet)")
logger.info("=" * 70)

rohf = RestrictedOpenShellHartreeFock(
    cgtos=cgtos_cat,
    nuclei=nuclei_cat,
    e_nuclear=e_nuclear_cat,
    n_closed=0,
    n_open=1,
).run()

logger.info(rohf)
logger.info("")
logger.info(f"  Converged          : {rohf.converged}")
logger.info(f"  Iterations         : {rohf.n_iterations}")
logger.info(f"  Electronic energy  : {rohf.e_electronic:.10f} Hartree")
logger.info(f"  Nuclear repulsion  : {e_nuclear_cat:.10f} Hartree")
logger.info(f"  Total energy       : {rohf.e_total:.10f} Hartree")
logger.info(f"  Stored matrices    : {sorted(rohf.matrices)}")
logger.info(f"  Orbital energies   : {rohf.matrices['epsilon']}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON: UHF vs ROHF FOR H2⁺
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Comparison: UHF vs ROHF for H2\u207a")
logger.info("=" * 70)

delta = abs(uhf.e_total - rohf.e_total)
logger.info(f"  UHF  total energy  : {uhf.e_total:.10f} Hartree")
logger.info(f"  ROHF total energy  : {rohf.e_total:.10f} Hartree")
logger.info(f"  |UHF - ROHF|       : {delta:.2e} Hartree")
logger.info(f"  Energies agree     : {delta < 1e-10}")
