# ── Example: Hartree-Fock initialization for a water molecule ────────
# Demonstrates RHF, UHF, and ROHF using mixed basis sets.

"""Example: RHF / UHF / ROHF initialization for a water molecule

This example demonstrates how to use :class:`RHF`, :class:`UHF`, and
:class:`ROHF` from ``q_block.theory.initialization`` to set up
Hartree-Fock calculations for a water molecule with a **mixed** basis
set (3-21G on H, 6-31G on O).

Three scenarios are shown:

1. **RHF** – Restricted Closed-Shell (neutral water, singlet).
2. **UHF** – Unrestricted (water cation H₂O⁺, doublet).
3. **ROHF** – Restricted Open-Shell (water cation H₂O⁺, doublet).

Steps for each scenario:
  1. Parse two Pople basis sets.
  2. Define atoms with coordinates, charges, and basis sets.
  3. Build a Molecule (GTO population happens automatically).
  4. Create an RHF / UHF / ROHF initialization.
  5. Inspect the resulting electron counts and basis-set size.
"""

from q_block.io.input_data import InputData
from q_block.io.basis_set import Pople
from q_block.systems.molecule import Molecule
from q_block.theory.initialization import RHF, UHF, ROHF

# ── 1. Create Pople basis set objects ────────────────────────────────
basis_3_21G = Pople(filepath="data/basis_set/gto_gaussian_format/3-21G.gbs")
basis_6_31G = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")


# =====================================================================
# Helper: build a water InputData with per-atom charges
# =====================================================================
def _make_water(charge_O: int = 0, charge_H1: int = 0,
                charge_H2: int = 0) -> InputData:
    """Return an :class:`InputData` for water with the given per-atom charges."""
    inp = InputData()
    inp.from_script(
        atom_data=[
            ["O",  0.0000,  0.0000,  0.1173, basis_6_31G, charge_O],
            ["H",  0.0000,  0.7572, -0.4692, basis_3_21G, charge_H1],
            ["H",  0.0000, -0.7572, -0.4692, basis_3_21G, charge_H2],
        ],
        atom_prefix="W",
    )
    return inp


# =====================================================================
# 2. RHF – Restricted Closed-Shell (neutral water, singlet)
# =====================================================================
print("=" * 64)
print("RHF – Restricted Closed-Shell  (neutral H₂O, singlet)")
print("=" * 64)

water_rhf = Molecule(input_data=_make_water(), multiplicity=1)
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

first_atom = rhf.molecule.atoms[0]
print(f"  First atom in Bohr       : {first_atom.coordinates}")


# =====================================================================
# 3. UHF – Unrestricted  (water cation H₂O⁺, doublet)
# =====================================================================
print()
print("=" * 64)
print("UHF – Unrestricted  (H₂O⁺  cation, doublet)")
print("=" * 64)

# Remove one electron from oxygen (charge_O = -1 → fewer electrons)
water_uhf = Molecule(input_data=_make_water(charge_O=-1), multiplicity=2)
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


# =====================================================================
# 4. ROHF – Restricted Open-Shell  (water cation H₂O⁺, doublet)
# =====================================================================
print()
print("=" * 64)
print("ROHF – Restricted Open-Shell  (H₂O⁺  cation, doublet)")
print("=" * 64)

# Same cation, but treated with ROHF (spin-pure)
water_rohf = Molecule(input_data=_make_water(charge_O=-1), multiplicity=2)
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


# =====================================================================
# 5. Comparison table
# =====================================================================
print()
print("=" * 64)
print("Comparison of HF methods for water")
print("=" * 64)
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
