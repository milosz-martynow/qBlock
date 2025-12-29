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

## Example: creating an AtomicSystem

Below is a minimal example showing how to create an :class:`AtomicSystem`
from an XYZ file or from a Python structure using the helpers in
``q_block.read_atoms``. The repository contains small example geometries in
``tests/verification_data/geometries`` (for instance ``water.xyz``).

```python
from q_block.read_atoms import populate_from_xyz_file, populate_from_script
from q_block.atomic_system import AtomicSystem

# Load from an XYZ file shipped with the tests
df = populate_from_xyz_file("tests/verification_data/geometries/water.xyz", atom_prefix="A")
system = AtomicSystem(df)
print(system)            # AtomicSystem(n_atoms=...)
print(system.atoms.head())

# Or build from a Python data structure: [symbol, x, y, z]
py_data = [["O", 0.000000, 0.000000, 0.000000], ["H", 0.758602, 0.000000, 0.504284], ["H", -0.758602, 0.000000, 0.504284]]
df2 = populate_from_script(py_data, atom_prefix="M")
system2 = AtomicSystem(df2)
print(system2.atoms)
```

Note: the example requires the ``pandas`` package to be installed in your
environment.
