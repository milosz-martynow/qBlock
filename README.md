# qBlock

A Python library for quantum chemistry calculations. Implemented numerical features:

- **SCF methods** — RHF, UHF, ROHF (Hartree-Fock); RKS, UKS, ROKS (Kohn-Sham DFT)
- **Basis sets** — GTO (Pople family, e.g. 3-21G, 6-311ppGss); each atom can carry its own independent basis set
- **Input** — geometry and basis set defined together per atom via `InputData.from_script`; supports both Å and Bohr
- **Integrals** — overlap, kinetic energy, nuclear attraction, two-electron repulsion (ERI)
- **XC functionals** — LDA (SVWN), GGA (PBE), hybrid (B3LYP)
- **Convergence** — DIIS acceleration for all SCF variants
- **Numerical integration** — Becke-partitioned grids for DFT quadrature

The following example shows how to set up and run an Unrestricted Hartree-Fock calculation on a water molecule using a mixed basis set. It demonstrates that basis sets are bundled with the package and can be assigned independently per atom. It also illustrates the typical qBlock pipeline: define input geometry, construct an atomic system (here `Molecule`), expand it into contracted GTOs, then choose and instantiate a solver — one of RHF, UHF, ROHF, RKS, UKS, or ROKS, passing the basis functions and electron counts — and call `run()`, which executes the SCF iterations internally and stores the converged energy and other results on the solver object.

```python
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.models.molecule import Molecule
from q_block.compute.environment.constants.numerical.pople import G321, G6311ppss
from q_block.compute.solvers.wavefunction.hartree_fock import UnrestrictedHartreeFock

# Build molecular geometry (coordinates in Å)
inp = InputData()
inp.from_script(atom_data=[
    ["O",  0.0000, 0.0000, 0.0000, G6311ppss],
    ["H",  0.7572, 0.0000, 0.5860, G321],
    ["H", -0.7572, 0.0000, 0.5860, G321],
])

# Construct the molecule and expand into contracted GTO basis functions
molecule = Molecule(input_data=inp, multiplicity=1)
molecule.to_bohr()
molecule.make_contracted_gaussian_type_orbital()

# Build UHF and run SCF on it.
uhf = UnrestrictedHartreeFock(
        cgtos=molecule.contracted_gaussian_type_orbitals, 
        n_alpha=5, 
        n_beta=5
    )
uhf.run()
print(f"Converged: {uhf.converged},  E_total = {uhf.e_total:.6f} Hartree")
```

The project is designed to be easy to understand, modify, and extend. The overall project follows the **LCT** (Learn-Compute-Tests) architecture, which organises the entire development workflow into three layers: `q_block/learn/` for requirements, documentation and examples, `q_block/compute/` for the core library, and `q_block/tests/` for correctness verification. The `q_block/compute/` layer internally follows the **MUSE** (Models-Utilities-Solvers-Environment) architecture, separating domain models, shared mathematics, numerical algorithms, and infrastructure concerns into distinct, independently navigable modules.

---

## Requirements
Software was developed and tested in below environment. 
- Windows 11 Pro
- Python 3.12

But, it does not means that it doesn't work in other environment.

## Installation

### Via PyPI (recommended)

The simplest way to install qBlock is directly from the Python Package Index (PyPI). This installs the package and all required runtime dependencies automatically.

```
pip install qBlock
```

To include optional extras:

```
pip install qBlock[test]      # adds pytest
pip install qBlock[dev]       # adds pylint, black, isort
pip install qBlock[all]       # adds all of the above
```

### Manual Installation

For development or to install directly from the [source on GitHub](https://github.com/milosz-martynow/qBlock), clone the repository and set up a virtual environment. Note that the GitHub repository may contain commits not yet released to PyPI, so versions between the two sources can differ.

```
python.exe -m venv .venv
.venv\Scripts\activate
```

Install qBlock with all dependencies:

```
pip.exe install -e .
```

If `setuptools` is missing:

```
pip.exe install setuptools==80.9.0
```

---

## Usage

### Run an example

Examples are standalone scripts located in `q_block/learn/examples/`:

```
python q_block/learn/examples/example_scf_hartree_fock.py
python q_block/learn/examples/example_overlap_integral.py
```

### Run tests

```
pytest.exe .
```

Run only verification or validation tests:

```
pytest.exe q_block/tests/verification/
pytest.exe q_block/tests/validation/
```

Some validation tests are computationally heavy — they run full SCF calculations across many atoms and molecules. To run a specific validation file or a single record, use pytest's `-k` flag. Atom test IDs follow the pattern `Z{atomic_number}_{symbol}`; molecule test IDs are their dictionary keys from `validation_data.py`:

```
pytest .\q_block\tests\validation\test_rhf.py -k "Z2_He"   # Helium atom only
pytest .\q_block\tests\validation\test_rhf.py -k "H2"      # H2 molecule only
```

### Format and lint

```
isort.exe .
black.exe --config=.blackrc .\q_block\ setup.py
pylint.exe --rcfile=.pylintrc .\q_block\ setup.py
```

---

## Architecture

The project - qBlock - is organized under the `q_block/` package into three sub-directories: `learn/`, `compute/`, and `tests/`. This order reflects the intended workflow - understand, implement, verify and after expansion looks as follows:

```
project/
└── q_block/                 # Top-level package
    ├── learn/               # Documentation, diagrams, and examples - no library code
    │   ├── architecture/    # Diagrams (e.g. UML) describing system structure and workflows
    │   └── examples/        # Runnable scripts demonstrating individual modules
    ├── compute/             # Core library
    │   ├── models/          # Domain layer: physical models and data structures
    │   ├── utilities/       # Shared mathematical functions used across the library
    │   ├── solvers/         # Algorithm layer: numerical solvers and convergence methods
    │   └── environment/     # Infrastructure: external data, I/O - input/output , and configuration
    │       ├── constants/   # Reference data: physical and numerical constants
    │       │   ├── natural/ # Nature based constants (e.g. physical and mathematical constants)
    │       │   └── numerical/ # Numerical parameters
    │       └── io/          # Interfaces for reading and writing data
    └── tests/               # Correctness verification at unit and system level
        ├── verification/    # Fine-grained per-module tests
        └── validation/      # End-to-end tests verified against known reference data
```
It is worth to highlight that 0'th level `learn/`, `compute/`, and `tests/` are verbs, whereas lower level elements are nouns. only environment/constants is divided into two two adjectives.

**Verification vs Validation:** *Verification* answers "Are we building the product right?" (implementation correctness). *Validation* answers "Are we building the right product?" (external requirements).

### Layer 1: `learn/`

The `learn/` directory is the entry point for understanding the project. It contains:

- **`requirements.md`** - Most important and "must have" file of this folder. Formal software requirements specification. Lists all numbered shall-statements that define what qBlock must implement, including supported theories, integrals, I/O formats, code style rules, and testing constraints. Also provides an acronym glossary.
- **`architecture/`** - UML diagrams (component, block definition, activity, use case) describing the system structure and workflows.
- **`examples/`** - Runnable Python scripts demonstrating how to use individual modules (integrals, SCF solvers, I/O, etc.).

`learn/` does not contain executable library code - it is purely for comprehension and exploration.

### Layer 2: `compute/`

The `compute/` directory is the core library. It contain all scripts and programs needed to fulfill `learn/requirements.md` requirements. Its internal architecture follows `MUSE` four layers:

```
compute/
├── models/          # Physical models and quantum theory
│   ├── integrals/       # Integral engines (overlap, kinetic, nuclear, ERI)
│   └── initialization/  # Calculation context, nuclear repulsion energy
├── utilities/       # Shared mathematical utilities
├── solvers/         # SCF algorithms (RHF, UHF, ROHF, DIIS)
└── environment/     # External data, I/O, configuration
    ├── constants/
    │   ├── natural/     # Physical constants, atomic data
    │   └── numerical/   # Numerical parameters, basis set files
    └── io/              # Input/output interfaces

```

- **`environment/`** - Infrastructure layer. Provides physical constants (`natural/`), numerical data and basis set files (`numerical/`), and I/O interfaces for reading coordinates, basis sets, and structured input/output data. Also manages runtime configuration.
- **`models/`** - Domain layer. Defines atomic and molecular data structures, GTO basis functions, quantum mechanical integrals, numerical integration grids, and the Hartree-Fock and DFT calculation contexts.
- **`solvers/`** - Algorithm layer. Implements SCF loop variants using the Template Method pattern. Hartree-Fock solvers (`RestrictedHartreeFock`, `UnrestrictedHartreeFock`, `RestrictedOpenShellHartreeFock`) and Kohn-Sham DFT solvers (`RestrictedKohnSham`, `RestrictedOpenShellKohnSham`, `UnrestrictedKohnSham`) inherit from an abstract `SCF` base. Exchange-correlation functionals (`SVWN`, `PBE`, `B3LYP`) provide the DFT energy and potential. Includes DIIS convergence acceleration and Becke-partitioned numerical integration grids.
- **`utilities/`** - Shared mathematical functions (Boys function, Hermite expansion, normalization, double factorial) used across models and solvers.

### Layer 3: `tests/`

The `tests/` directory is an place for storing all methods, proofs and checckers to test that software from `compute/` fulfils `lear/requirements.md` requirements. The `tests/` directory verifies correctness at two levels:

```
tests/
├── verification/  # Fine-grained per-module tests
│   ├── models/
│   ├── solvers/
│   ├── environment/
│   ├── utilities/
│   └── verification_data/   # Golden reference JSON files
└── validation/    # End-to-end HF and DFT calculations
    ├── test_rhf.py
    ├── test_uhf.py
    ├── test_rohf.py
    ├── test_rks.py
    ├── test_roks.py
    ├── test_uks.py
    ├── validation_data.py   # Reference ionization energies
    └── templates.py         # Reusable test factories
```

*Verification* answers "Are we building the product right?" (implementation correctness). *Validation* answers "Are we building the right product?" (external requirements).

- **`verification/`** - Tests individual classes and functions in isolation. Uses `@pytest.mark.parametrize` extensively; no test classes. References used for verification testing are stored as JSON in `verification_data/`.
- **`validation/`** - End-to-end tests that run full SCF calculations (HF and DFT) on atoms and molecules and verify results against known ionization energies (Koopmans theorem) and physical bounds within a defined tolerance.

## Development Note

This repository was developed with the assistance of Large Language Models (LLMs). However, every requirement, architectural decision, algorithm, and implementation detail was individually analysed, reviewed, and explicitly defined by the author. LLMs served as a productivity tool; all intellectual and domain-specific content reflects the author's own understanding and judgment.

