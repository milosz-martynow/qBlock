"""Example: creating and inspecting a Molecule

Below is a minimal example showing how to create a `Molecule`
from an XYZ file or from a Python structure using the `InputData`
helper class from `q_block.io.input_data`. The repository contains small
example geometries in `tests/verification_data/geometries` (for instance
`water.xyz`).
"""

from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule

# Load atomic coordinates for a water molecule from an XYZ file
water_input = InputData()
water_input.from_xyz_file(
    xyz_path="tests/verification_data/geometries/water.xyz",
    atom_prefix="A",
)

# Build a Molecule from InputData
water_molecule = Molecule(input_data=water_input)
print(water_molecule)
print(water_molecule.input_data.atoms.head())

# Or build a water-like molecule from a Python data structure: [symbol, x, y, z]
water_script_atoms = [
    ["O", 0.000000, 0.000000, 0.000000],
    ["H", 0.758602, 0.000000, 0.504284],
    ["H", -0.758602, 0.000000, 0.504284],
]
water_input_from_script = InputData()
water_input_from_script.from_script(
    atom_data=water_script_atoms,
    atom_prefix="M",
)
water_molecule_from_script = Molecule(input_data=water_input_from_script)
print(water_molecule_from_script.input_data.atoms)
