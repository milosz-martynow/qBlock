"""
Example: Creating and Inspecting a Molecule
============================================

This script demonstrates how to create a Molecule from a Python structure
using the InputData helper class.

Steps:
    1. Create InputData with atom definitions
    2. Build a Molecule from InputData
    3. Inspect molecule properties
"""

from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEFINE ATOMS USING InputData
# ══════════════════════════════════════════════════════════════════════════════
# InputData.from_script() accepts a list of atom definitions.
# Each entry is: [symbol, x, y, z]
# Coordinates are in Ångströms.

water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O",  0.000000,  0.000000,  0.000000],   # Oxygen at origin
        ["H",  0.758602,  0.000000,  0.504284],   # Hydrogen 1
        ["H", -0.758602,  0.000000,  0.504284],   # Hydrogen 2
    ],
    atom_prefix="W",  # Prefix for atom IDs (W1, W2, W3)
)

print("=== InputData atoms DataFrame ===")
print(water_input.atoms)
print()


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD MOLECULE FROM InputData
# ══════════════════════════════════════════════════════════════════════════════
# The Molecule class wraps InputData and provides molecular properties.
# It extracts Atom instances and computes aggregate quantities.

water = Molecule(input_data=water_input)

print("=== Molecule object ===")
print(water)
print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. INSPECT MOLECULE PROPERTIES
# ══════════════════════════════════════════════════════════════════════════════

print("=== Molecule properties ===")
print(f"  Number of atoms:     {len(water.atoms)}")
print(f"  Total atomic number: {water.total_atomic_number}")
print(f"  Total electrons:     {water.n_electrons}")
print(f"  Charge:              {water.charge}")
print(f"  Multiplicity:        {water.multiplicity}")
print()

# Access individual atoms
print("=== Individual atoms ===")
for i, atom in enumerate(water.atoms):
    print(f"  Atom {i}: {atom}")
