# Dependencies

## Operational system
Operating System in which `qBlock` developed is Windows 11 Pro. For this environment this `README.md` is written.
Nevertheless, it should be not a problem to follow below command with small changes to run it under Linux OS.

## Python 
Python in version [3.12.7](https://peps.python.org/pep-0693/). 

For better maintenance of Python code it is worth to use 
[Python Virtual Environment](https://docs.python.org/3/library/venv.html).

You can create Python Virtual Environment by typing in terminal of your project root folder:
```
python.exe -m venv .venv
.venv\Scripts\activate
```
Note - `'.venv'` name is included in `.gitignore` file.

### PIP
Upgrading pip will be useful, when issues with requirements libraries araises:
```
python.exe -m pip install --upgrade pip
```

## Install packages
To install `qBlock` with all project dependencies:
```
pip.exe install -e .
```

Sometimes, python does not come with `setuptools`. If so - above will not work until
`setuptools` will be installed virtual environment:
```
pip.exe install setuptools==80.9.0
```
## Test and formatting:
To test and format code type in terminal:
```
isort.exe .
black.exe --config=.blackrc .\q_block\ .\tests\ setup.py
pylint.exe --rcfile=.pylintrc .\q_block\ .\tests\ setup.py
pytest.exe .
```

## Example: creating and inspecting a Molecule

Below is a minimal example showing how to create a :class:`Molecule`
from an XYZ file or from a Python structure using the :class:`InputData`
helper class from ``q_block.input_data``. The repository contains small
example geometries in ``tests/verification_data/geometries`` (for instance
``water.xyz``).

```python
from q_block.input_data import InputData
from q_block.molecule import Molecule

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
```

## Example: water molecule and spin-orbital DataFrame

The snippet below shows how to construct a :class:`Molecule` from an XYZ
geometry, attach different Pople basis sets to hydrogen and oxygen atoms,
and then export the occupied spin-orbitals into a single
:class:`pandas.DataFrame` using :meth:`Molecule.to_dataframe`.

```python
from pathlib import Path
import pandas as pd

from q_block.input_data import InputData
from q_block.molecule import Molecule
from q_block.basis_set_pople import parse_gaussian_basis

# Work from the project root directory
project_root = Path.cwd()

xyz_path = project_root / "tests" / "verification_data" / "geometries" / "water.xyz"
basis_3_21G_path = project_root / "data" / "basis_set" / "gto_gaussian_format" / "3-21G.gbs"
basis_6_31G_path = project_root / "data" / "basis_set" / "gto_gaussian_format" / "6-31G.gbs"

# 1. Load geometry into InputData
input_data = InputData()
input_data.from_xyz_file(xyz_path=xyz_path)

# 2. Load Pople basis sets
basis_3_21G = parse_gaussian_basis(filepath=str(basis_3_21G_path))
basis_6_31G = parse_gaussian_basis(filepath=str(basis_6_31G_path))

# 3. Attach basis sets to atoms: 3-21G for H, 6-31G for O
for row in input_data.atoms.itertuples():
    atom = row.atom
    symbol = row.symbol

    if symbol == "H":
        atom.basis_set = basis_3_21G["H"]
    elif symbol == "O":
        atom.basis_set = basis_6_31G["O"]

# 4. Build the Molecule; spin-orbitals are populated with GTO data
water = Molecule(input_data=input_data)

# 5. Export occupied spin-orbital data to a multi-index DataFrame
spinorb_df = water.to_dataframe()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(spinorb_df)
```
