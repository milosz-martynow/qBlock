# qBlock

A Python library for quantum chemistry calculations. qBlock implements Self-Consistent Field (SCF) methods for both Hartree-Fock (HF) — Restricted (RHF), Unrestricted (UHF), and Restricted Open-Shell (ROHF) — and Kohn-Sham Density Functional Theory (DFT) with LDA (SVWN), GGA (PBE), and hybrid (B3LYP) exchange-correlation functionals, built on Gaussian-type orbital (GTO) basis sets.

The project is designed to be easy to understand, modify, and extend. The overall project follows the **LCT** (Learn-Compute-Tests) architecture, which organises the entire development workflow into three layers: `learn/` for requirements, documentation and examples, `compute/` for the core library, and `tests/` for correctness verification. The `compute/` layer internally follows the **MUSE** (Models-Utilities-Solvers-Environment) architecture, separating domain models, shared mathematics, numerical algorithms, and infrastructure concerns into distinct, independently navigable modules.

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

Examples are standalone scripts located in `learn/examples/`:

```
python learn/examples/example_scf_hartree_fock.py
python learn/examples/example_overlap_integral.py
```

### Run tests

```
pytest.exe .
```

Run only verification or validation tests:

```
pytest.exe tests/verification/
pytest.exe tests/validation/
```

### Format and lint

```
isort.exe .
black.exe --config=.blackrc .\compute\ .\tests\ setup.py
pylint.exe --rcfile=.pylintrc .\compute\ .\tests\ setup.py
```

---

## Architecture

The project - qBlock - is organized into three top-level directories: `learn/`, `compute/`, and `tests/`. This order reflects the intended workflow - understand, implement, verify and after expansion looks as follows:

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
│       └── io/              # Interfaces for reading and writing data
└── tests/                   # Correctness verification at unit and system level
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
- *(Future)* Full API and mathematical documentation.

`learn/` does not contain executable library code - it is purely for comprehension and exploration.

### Layer 2: `compute/`

The `compute/` directory is the core library. It contain all scripts and programs needed to fulfill `learn/requirements.md` requirements. Its internal architecture follows `MUSE` four layers:

```
compute/
├── models/          # Physical models and quantum theory
│   ├── integrals/       # Integral engines (overlap, kinetic, nuclear, ERI)
│   └── initialization/  # Calculation context, nuclear repulsion energy
├─ utilities/       # Shared mathematical utilities
├── solvers/         # SCF algorithms (RHF, UHF, ROHF, DIIS)
└── environment/     # External data, I/O, configuration
    ├── constants/
    │   ├── natural/     # Physical constants, atomic data
    │   └── numerical/   # Numerical parameters, basis set files
    └── io/              # Input/output interfaces

```

- **`environment/`** - Infrastructure layer. Provides physical constants (`natural/`), numerical data and basis set files (`numerical/`), and I/O interfaces for reading coordinates, basis sets, and structured input/output data. Also manages runtime configuration.
- **`models/`** - Domain layer. Defines atomic and molecular data structures, GTO basis functions, quantum mechanical integrals, numerical integration grids, and the Hartree-Fock and DFT calculation contexts.
- **`solvers/`** - Algorithm layer. Implements SCF loop variants using the Template Method pattern. Hartree-Fock solvers (`RestrictedHartreeFock`, `UnrestrictedHartreeFock`, `RestrictedOpenShellHartreeFock`) and Kohn-Sham DFT solvers (`RestrictedKohnSham`, `UnrestrictedKohnSham`) inherit from an abstract `SCF` base. Exchange-correlation functionals (`SVWN`, `PBE`, `B3LYP`) provide the DFT energy and potential. Includes DIIS convergence acceleration and Becke-partitioned numerical integration grids.
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
└── validation/    # End-to-end HF calculations
    ├── test_rhf.py
    ├── test_uhf.py
    ├── test_rohf.py
    ├── validation_data.py   # Reference ionization energies
    └── templates.py         # Reusable test factories
```

*Verification* answers "Are we building the product right?" (implementation correctness). *Validation* answers "Are we building the right product?" (external requirements).

- **`verification/`** - Tests individual classes and functions in isolation. Uses `@pytest.mark.parametrize` extensively; no test classes. References used for verification testing are stored as JSON in `verification_data/`.
- **`validation/`** - End-to-end tests that run full SCF calculations (HF and DFT) on atoms and molecules and verify results against known ionization energies (Koopmans theorem) and physical bounds within a defined tolerance.

---

## Development Note

This repository was developed with the assistance of Large Language Models (LLMs). However, every requirement, architectural decision, algorithm, and implementation detail was individually analysed, reviewed, and explicitly defined by the author. LLMs served as a productivity tool; all intellectual and domain-specific content reflects the author's own understanding and judgment.

