# qBlock

A Python library for quantum chemistry calculations. qBlock implements Hartree-Fock self-consistent field (SCF) methods - Restricted (RHF), Unrestricted (UHF), and Restricted Open-Shell (ROHF) - built on Gaussian-type orbital (GTO) basis sets.

---

## Requirements

- Windows 11 Pro (Linux should work with minor command adjustments)
- Python 3.12

### Setup

Create and activate a virtual environment:

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

Examples are standalone scripts located in `learn/examples/`:

```
python learn/examples/example_scf_hartree_fock.py
python learn/examples/example_overlap_integral.py
```

### Run tests

```
pytest.exe .
```

Run only unit or validation tests:

```
pytest.exe tests/unit_tests/
pytest.exe tests/validation_tests/
```

### Format and lint

```
isort.exe .
black.exe --config=.blackrc .\compute\ .\tests\ setup.py
pylint.exe --rcfile=.pylintrc .\compute\ .\tests\ setup.py
```

---

## Architecture

The project is organized into three top-level directories: `learn/`, `compute/`, and `tests/`. This order reflects the intended workflow - understand, implement, verify. Project follows LCT generic architecture which is described in detail in further sections:

```
project/
├── learn/                   # Documentation, diagrams, and examples - no library code
│   ├── architecture/        # Diagrams (e.g. UML) describing system structure and workflows
│   └── examples/            # Runnable scripts demonstrating individual modules
├── compute/                 # Core library
│   ├── models/              # Domain layer: physical models and data structures
│   ├── utilities/           # Shared mathematical functions used across the library
│   ├── solvers/             # Algorithm layer: numerical solvers and convergence methods
│   └── environment/         # Infrastructure: external data, I/O - input/output , and configuration
│       ├── constants/       # Reference data: physical and numerical constants
│       │   ├── natural/     # Nature based constants (e.g. physical and mathematical constants)
│       │   └── numerical/   # Numerical parameters
│       └── io/              # Interfaces for reading coordinates, basis sets, and data
└── tests/                   # Correctness verification at unit and system level
    ├── unit_tests/          # Fine-grained per-module tests
    └── validation_tests/    # End-to-end tests verified against known reference data
```

### `learn/`

The `learn/` directory is the entry point for understanding the project. It contains:

- **`architecture/`** - UML diagrams (component, block definition, activity, use case) describing the system structure and workflows.
- **`examples/`** - Runnable Python scripts demonstrating how to use individual modules (integrals, SCF solvers, I/O, etc.).
- *(Future)* Full API and mathematical documentation.

`learn/` does not contain executable library code - it is purely for comprehension and exploration.

### `compute/`

The `compute/` directory is the core library. Its internal architecture follows four layers:

```
compute/
├── environment/     # External data, I/O, configuration
│   ├── constants/
│   │   ├── natural/     # Physical constants, atomic data
│   │   └── numerical/   # Numerical parameters, basis set files
│   └── io/              # Input/output interfaces
├── models/          # Physical models and quantum theory
│   ├── integrals/       # Integral engines (overlap, kinetic, nuclear, ERI)
│   └── initialization/  # Calculation context, nuclear repulsion energy
├── solvers/         # SCF algorithms (RHF, UHF, ROHF, DIIS)
└── utilities/       # Shared mathematical utilities
```

- **`environment/`** - Infrastructure layer. Provides physical constants (`natural/`), numerical data and basis set files (`numerical/`), and I/O interfaces for reading coordinates, basis sets, and structured input/output data. Also manages runtime configuration.
- **`models/`** - Domain layer. Defines atomic and molecular data structures, GTO basis functions, quantum mechanical integrals, and the Hartree-Fock calculation context.
- **`solvers/`** - Algorithm layer. Implements SCF loop variants using the Template Method pattern. Each solver (`RestrictedHartreeFock`, `UnrestrictedHartreeFock`, `RestrictedOpenShellHartreeFock`) inherits from an abstract `SCF` base and provides its own Fock build, density matrix, and energy routines. Includes DIIS convergence acceleration.
- **`utilities/`** - Shared mathematical functions (Boys function, Hermite expansion, normalization, double factorial) used across models and solvers.

### `tests/`

The `tests/` directory verifies correctness at two levels:

```
tests/
├── unit_tests/          # Fine-grained per-module tests
│   ├── models/
│   ├── solvers/
│   ├── environment/
│   ├── utilities/
│   └── verification_data/   # Golden reference JSON files
└── validation_tests/    # End-to-end HF calculations
    ├── test_rhf.py
    ├── test_uhf.py
    ├── test_rohf.py
    ├── validation_data.py   # Reference ionization energies
    └── templates.py         # Reusable test factories
```

- **`unit_tests/`** - Tests individual classes and functions in isolation. Uses `@pytest.mark.parametrize` extensively; no test classes. Reference outputs are stored as JSON in `verification_data/`.
- **`validation_tests/`** - End-to-end tests that run full SCF calculations on atoms and molecules and verify results against known ionization energies (Koopmans theorem) within a defined tolerance.