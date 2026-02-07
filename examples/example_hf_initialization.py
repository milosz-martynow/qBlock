# ── Example: Hartree-Fock initialization for a water molecule ────────
# Mixed basis set: 3-21G for H atoms, 6-31G for O atom.

"""Example: Hartree-Fock initialization for a water molecule

This example demonstrates how to use `HartreeFock` from
`q_block.theory.initialization` to set up an RHF calculation for
a water molecule using a **mixed** basis set (3-21G on H, 6-31G on O).

Steps:
  1. Parse two Pople basis sets.
  2. Define atoms with coordinates and basis sets in a single list.
  3. Build a Molecule (GTO population happens automatically).
  4. Create a HartreeFock initialization.
  5. Inspect the resulting electron counts and basis-set size.
"""

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.initialization import HartreeFock

# ── 1. Create Pople basis set objects ────────────────────────────────
basis_3_21G = Pople(filepath="data/basis_set/gto_gaussian_format/3-21G.gbs")
basis_6_31G = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")

# ── 2. Define atoms: [symbol, x, y, z, basis_set] ───────────────────
water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O",  0.0000,  0.0000,  0.1173, basis_6_31G],
        ["H",  0.0000,  0.7572, -0.4692, basis_3_21G],
        ["H",  0.0000, -0.7572, -0.4692, basis_3_21G],
    ],
    atom_prefix="W",
)

# ── 3. Build the Molecule (GTO population automatic) ─────────────────
water = Molecule(input_data=water_input)

# ── 4. Hartree-Fock initialization (RHF, mixed basis) ───────────────
hf = HartreeFock(
    molecule=water,
    charge=0,
    multiplicity=1,
    hf_method="RHF",
)

# ── 5. Inspect results ──────────────────────────────────────────────
print(hf)
print()

print(f"Total electrons : {hf.n_electrons}")
print(f"Occupied orbitals (n_occ): {hf.n_occ}")
print(f"Alpha electrons  : {hf.n_alpha}")
print(f"Beta electrons   : {hf.n_beta}")
print(f"Basis functions  : {hf.n_basis}")
print(f"HF method        : {hf.hf_method}")

# Coordinates are now in Bohr (converted in-place during initialization)
first_atom = hf.molecule.atoms[0]
print(f"\nFirst atom in Bohr: {first_atom.coordinates}")
