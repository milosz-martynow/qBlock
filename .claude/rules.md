# Coding Rules

## Language & Runtime
- Python 3.12 only. 
- No PEP 604 union syntax (`X | Y`); use `typing.Optional`, `typing.Union`.
- Dependencies: numpy, scipy, pandas, pytest. No numba, cython, or C extensions.

## Formatting
- **Black**: line-length 84, target `py312` (config in `.blackrc`).
- **Pylint**: max-line-length 84 (config in `.pylintrc`).
- **isort** for import ordering.
- Run before commit: `isort . && black --config=.blackrc . && pylint --rcfile=.pylintrc .`

## Naming
- Classes should be Noun in `PascalCase`. Examples:  `ContractedGaussianTypeOrbital`, `NuclearAttraction`.
- Functions/methods should be Verb in `snake_case`. Examples: `_build_fock`, `make_contracted_gaussian_type_orbital`.
- Constants: `UPPER_SNAKE_CASE` — `HARTREE_TO_EV`, `ATOMS_SYMBOLS_Z_TO_SYMBOL`.
- Private members: single `_` prefix.

## Imports
- Absolute imports in source (`from q_block.methods.diis import DIIS`).
- Imports should be defined on the top of the script.
- Imports should not be defined inside class or function.
- Relative imports only inside `__init__.py` re-exports.
- Order: stdlib → third-party → project.
- If import is not used in a given script, then it should not be attached to this script.

## Type Annotations
- All methods, functions, tests, classes and `__init__` fully annotated.
- If nothing is returned by method or function rtype should follow `-> None`.
- Use `typing` module generics: `List`, `Dict`, `Tuple`, `Optional`.
- Annotate class-level attributes (`n_basis: int`, `matrix: np.ndarray`).

## Docstrings
- Sphinx/reST style: `:param name:`, `:type name:`, `:returns:`, `:rtype:`, `:raises:`.
- Use `r"""` raw strings for math blocks (`.. math::`).
- Class docstrings: Sphinx params for `__init__`, NumPy-style `Attributes` section for class attrs.

## Tests
- No test classes. 
- Bare test functions with `@pytest.mark.parametrize`.
- Unit tests mirror source tree: `tests/unit_tests/io/test_coordinates.py` ↔ `q_block/io/coordinates.py`.
- Validation tests use factory functions from `templates.py`.
- Test IDs: `Z{atomic_number}_{symbol}` for atoms.
- All unit tests should be run to test applied changes.
- One validation test for small atom and one validation test for small molecule should be run to test applied chandes

## Architecture
- Keep algorithmic optimizations pure Python (caching, DP tables, screening). No compiled extensions.

## Don'ts
- Don't add features or refactor beyond what's asked.
- Don't create helper abstractions for one-off operations.
- Don't modify public API exports in `__init__.py` without explicit request.
