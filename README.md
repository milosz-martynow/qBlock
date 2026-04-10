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

## Test and formatting
To test and format code type in terminal:
```
isort.exe .
black.exe --config=.blackrc .\q_block\ .\tests\ setup.py
pylint.exe --rcfile=.pylintrc .\q_block\ .\tests\ setup.py
pytest.exe .
```

## Examples

Code examples demonstrating how to use the library are provided as runnable
Python scripts in the `examples` directory in the project root.

## Environment Module

### Overview

The `environment/` module provides the foundational infrastructure for qBlock calculations, consolidating all external data sources, configuration, and I/O interfaces into a single cohesive package.

### Structure

```
q_block/environment/
├── __init__.py              # Module exports
├── configuration.py         # Configuration management
├── constants/              # Natural and numerical constants
│   ├── natural/           # Atomic data, physical constants
│   └── numerical/         # Basis sets, numerical parameters
└── io/                    # Input/output interfaces
    ├── basis_set.py       # Basis set readers (Pople, etc.)
    ├── coordinates.py     # Coordinate systems
    ├── input_data.py      # Input data container
    └── output_data.py     # Output data container
```

### Key Components

#### 1. **constants/**
Provides fundamental data for quantum chemistry calculations:
- **natural/**: Atomic data (symbols, masses, electron configurations)
- **numerical/**: Basis set files (.gbs format) and numerical parameters

#### 2. **io/**
Input/output interfaces for data exchange:
- `BasisSet` classes for reading Gaussian basis sets
- `Coordinates` for spatial representations (Cartesian, etc.)
- `InputData` for structured molecular input
- `OutputData` for calculation results with iteration history

#### 3. **configuration.py**
Centralized configuration management:
- `PathConfiguration`: File paths for inputs and outputs
- `CalculationDefaults`: SCF and convergence parameters
- `OutputSettings`: Output format preferences and filename prefix

**Output Prefix Feature**: All saved output files can use a configurable prefix (default: `"output"`). This helps organize results from multiple calculations:
```python
config.output.output_prefix = "h2_molecule"
# Files will be named: h2_molecule_results.json, h2_molecule_output.txt, etc.
```

### Configuration Management

#### Default Configuration File

qBlock automatically searches for configuration in the project root:
1. `.qblock.config` (recommended)
2. `qblock.config.json`
3. `.config` (fallback)

#### Usage Examples

**1. Use default configuration:**
```python
from q_block.environment import Configuration

config = Configuration()
```

**2. Load from file:**
```python
config = Configuration.from_file(".qblock.config")
# Or auto-detect:
config = Configuration.from_file()
```

**3. Load from environment variables:**
```python
# Set: QBLOCK_OUTPUT_DIR, QBLOCK_MAX_SCF_ITERATIONS, etc.
config = Configuration.from_env()
```

**4. Override specific settings:**
```python
config = Configuration(
    output_dir="custom_output",
    max_scf_iterations=200,
)
```

#### Configuration Priority

Settings are applied in this order (later overrides earlier):
1. **System defaults** — Built-in sensible defaults
2. **Config file** — `.qblock.config` or specified file
3. **Environment variables** — `QBLOCK_*` variables
4. **Constructor arguments** — Direct parameter overrides

### Example Configuration File

```json
{
  "paths": {
    "output_dir": "output",
    "log_dir": "logs"
  },
  "calculation": {
    "max_scf_iterations": 100,
    "scf_convergence_threshold": 1e-08
  },
  "output": {
    "save_json": true,
    "log_level": "INFO",
    "output_prefix": "output"
  }
}
```

### Migration from Previous Structure

The `environment/` module consolidates what were previously separate top-level modules:

| Previous | Current |
|----------|---------|
| `q_block.constants` | `q_block.environment.constants` |
| `q_block.io` | `q_block.environment.io` |
| `q_block.configuration` | `q_block.environment.configuration` |

**Convenience imports** are available through `q_block.__init__.py` for backward compatibility:

```python
# Both work:
from q_block.environment.io import InputData
from q_block import InputData  # Re-exported
```

### Design Rationale

Grouping `constants`, `io`, and `configuration` into `environment` provides:

1. **Semantic clarity**: All components manage external data and runtime environment
2. **Reduced coupling**: Clear boundary between infrastructure and computation
3. **Easier maintenance**: Related functionality in one place
4. **Explicit dependencies**: Solvers and models explicitly depend on environment

### Testing

All tests have been updated to use the new import paths:

```bash
# Test configuration
pytest tests/unit_tests/environment/test_configuration.py

# Test I/O modules
pytest tests/unit_tests/environment/io/

# Run all tests
pytest tests/
```
