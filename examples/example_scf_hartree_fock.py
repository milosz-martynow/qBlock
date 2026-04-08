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

import numpy as np

from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule
from q_block.theory.initialization import RHF, UHF, ROHF
from q_block.theory.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.solvers.wavefunction.hartree_fock import (
    RestrictedHartreeFock,
    UnrestrictedHartreeFock,
    RestrictedOpenShellHartreeFock,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SET
# ══════════════════════════════════════════════════════════════════════════════

basis_3_21G = Pople(filepath="data/basis_set/pople/3-21G.gbs")
print("Loaded basis set: 3-21G\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: build molecule and extract SCF inputs
# ══════════════════════════════════════════════════════════════════════════════

def prepare_molecule(atom_data, multiplicity=1):
    """Build a Molecule and return (cgtos, nuclei, e_nuclear).

    :param atom_data: List of [symbol, x, y, z, basis_set, charge] entries.
    :type atom_data: list
    :param multiplicity: Spin multiplicity (2S+1).
    :type multiplicity: int
    :returns: (cgtos, nuclei, e_nuclear) ready for HF constructors.
    :rtype: tuple
    """
    inp = InputData()
    inp.from_script(atom_data=atom_data)
    mol = Molecule(input_data=inp, multiplicity=multiplicity)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    cgtos = mol.contracted_gaussian_type_orbitals
    nuclei = [
        (atom.atomic_number,
         (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z))
        for atom in mol.atoms
    ]
    e_nuclear = NuclearRepulsionEnergy(mol).energy
    return cgtos, nuclei, e_nuclear


# ══════════════════════════════════════════════════════════════════════════════
# 2. RHF – RESTRICTED CLOSED-SHELL ON H2 (SINGLET)
# ══════════════════════════════════════════════════════════════════════════════
# H2 has 2 electrons -> 1 doubly-occupied orbital -> singlet.

print("=" * 70)
print("RHF – Restricted Closed-Shell (H2, singlet)")
print("=" * 70)

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

print(rhf)
print()
print(f"  Converged          : {rhf.converged}")
print(f"  Iterations         : {rhf.n_iterations}")
print(f"  Electronic energy  : {rhf.e_electronic:.10f} Hartree")
print(f"  Nuclear repulsion  : {e_nuclear:.10f} Hartree")
print(f"  Total energy       : {rhf.e_total:.10f} Hartree")
print(f"  Stored matrices    : {sorted(rhf.matrices)}")
print(f"  Orbital energies   : {rhf.matrices['epsilon']}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. UHF – UNRESTRICTED ON H2⁺ (DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# H2⁺ has 1 electron -> doublet (one unpaired alpha electron).
# Charge is set on one hydrogen atom (charge=-1 removes one electron).

print("=" * 70)
print("UHF – Unrestricted (H2⁺ cation, doublet)")
print("=" * 70)

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

print(uhf)
print()
print(f"  Converged          : {uhf.converged}")
print(f"  Iterations         : {uhf.n_iterations}")
print(f"  Electronic energy  : {uhf.e_electronic:.10f} Hartree")
print(f"  Nuclear repulsion  : {e_nuclear_cat:.10f} Hartree")
print(f"  Total energy       : {uhf.e_total:.10f} Hartree")
print(f"  Stored matrices    : {sorted(uhf.matrices)}")
print(f"  Alpha orb. energies: {uhf.matrices['epsilon_alpha']}")
print(f"  Beta  orb. energies: {uhf.matrices['epsilon_beta']}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROHF – RESTRICTED OPEN-SHELL ON H2⁺ (DOUBLET)
# ══════════════════════════════════════════════════════════════════════════════
# Same H2⁺ system, but with shared spatial orbitals and the Roothaan
# effective Fock matrix.  n_closed=0 (no doubly-occupied), n_open=1.

print("=" * 70)
print("ROHF – Restricted Open-Shell (H2⁺ cation, doublet)")
print("=" * 70)

rohf = RestrictedOpenShellHartreeFock(
    cgtos=cgtos_cat,
    nuclei=nuclei_cat,
    e_nuclear=e_nuclear_cat,
    n_closed=0,
    n_open=1,
).run()

print(rohf)
print()
print(f"  Converged          : {rohf.converged}")
print(f"  Iterations         : {rohf.n_iterations}")
print(f"  Electronic energy  : {rohf.e_electronic:.10f} Hartree")
print(f"  Nuclear repulsion  : {e_nuclear_cat:.10f} Hartree")
print(f"  Total energy       : {rohf.e_total:.10f} Hartree")
print(f"  Stored matrices    : {sorted(rohf.matrices)}")
print(f"  Orbital energies   : {rohf.matrices['epsilon']}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPARISON: UHF vs ROHF FOR H2⁺
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Comparison: UHF vs ROHF for H2⁺")
print("=" * 70)

delta = abs(uhf.e_total - rohf.e_total)
print(f"  UHF  total energy  : {uhf.e_total:.10f} Hartree")
print(f"  ROHF total energy  : {rohf.e_total:.10f} Hartree")
print(f"  |UHF - ROHF|       : {delta:.2e} Hartree")
print(f"  Energies agree     : {delta < 1e-10}")
print()
