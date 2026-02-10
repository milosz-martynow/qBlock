"""Example: creating and inspecting a Molecule

Below is a minimal example showing how to create a `Molecule`
from a Python structure using the `InputData` helper class from
`q_block.io.input_data`.
"""

from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule

# Define atoms inline: [symbol, x, y, z]
water_input = InputData()
water_input.from_script(
    atom_data=[
        ["O",  0.000000,  0.000000,  0.000000],
        ["H",  0.758602,  0.000000,  0.504284],
        ["H", -0.758602,  0.000000,  0.504284],
    ],
    atom_prefix="W",
)

# Build a Molecule from InputData
water = Molecule(input_data=water_input)
print(water)
print(water.input_data.atoms)
