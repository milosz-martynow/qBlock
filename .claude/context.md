# Project Context

**qBlock** — Electronic structure library in Python 3.12. Author: Miłosz Martynow. Implements Hartree-Fock (RHF, UHF, ROHF) via McMurchie-Davidson integrals and SCF with DIIS acceleration.

## Package Layers
- `environment/` Foundational infrastructure: configuration, logging, constants (natural/numerical), I/O interfaces.
- `models/` Physical models: particles, orbitals, composite systems (atoms, molecules, crystals), basis functions, integral engines, initialization.
- `solvers/` Numerical algorithms: SCF loop, DIIS, diagonalisation, HF variants (RHF, UHF, ROHF), KS-DFT variants (RKS, UKS, ROKS), XC functionals (SVWN, PBE, B3LYP).
- `utilities/` Pure-math helpers: Boys function, double factorial, normalization constants, Hermite tables.

## Root Files
- `setup.py` — Package definition.
- `README.md` — Setup instructions and project description.

## Other Directories
- `q_block/compute/environment/constants/numerical/basis_set/pople/` — Gaussian basis set files like: `STO-3G`, `3-21G`.
- `examples/` — runnable scripts demonstrating the API (molecule creation, integrals, SCF, GTO dataframe), located under `q_block/learn/examples/`.
- `q_block/learn/documentation/` — BDD and component diagrams for each sub-package (`environment`, `models`, `solvers`, `utilities`).

## Key Data Flow (SCF)
1. Load basis (`.gbs` file) → `Pople`
2. Build `InputData` → `Molecule(input_data, multiplicity, charge)`
3. `mol.to_bohr()` → `mol.make_contracted_gaussian_type_orbital()`
4. Extract CGTOs, nuclei list, nuclear repulsion energy
5. Instantiate HF variant → `.run()` → converged results with orbital energies

## How to Run
```powershell
# Setup
python -m venv .venv; .venv\Scripts\activate
pip install -e .

# Test & lint
isort .
black --config=.blackrc .\q_block\ setup.py
pylint --rcfile=.pylintrc .\q_block\ setup.py
pytest .
```

## Test Suites

**Verification** ("Are we building the product right?") tests implementation correctness. **Validation** ("Are we building the right product?") tests against external requirements.

### Verification Tests (`q_block/tests/verification/`)
Per-module, parametrized, no test classes — bare functions with `@pytest.mark.parametrize`.
- `constants.py` — shared fixtures: pre-loaded basis sets (`BASIS_3_21G`, `BASIS_STO_3G`, …), path constants (`BASIS_ROOT`, `GOLDEN_ROOT`, `GEOMETRIES_DIR`), `ORIGIN` coordinate.
- `utilities/` — directory mirroring `compute/utilities/`; `__init__.py` provides shared helpers: `ALL_ORBITAL_COMPONENTS`, `ORBITAL_LABELS`, `orbital_id()`, `get_orbital_ids()`.
- `verification_data/` — golden reference files: `expected_atom_empirical.py`, `expected_atom_pure.py`, `gto_population/*.json`, `geometries/*.xyz` (H2, water, azobenzene, tetraethylammonium), `tools/` (generator scripts).
- Mirror structure: `q_block/tests/verification/environment/io/test_coordinates.py` ↔ `q_block/compute/environment/io/coordinates.py`, etc.

### Validation Tests (`q_block/tests/validation/`)
End-to-end HF calculations verifying Koopmans' theorem ionization energies.
- `validation_data.py` — reference dicts (`ATOMS`, `MOLECULES`) mapping Z (1–54) / molecule name → `{symbol, config, multiplicity, n_electrons, n_alpha, n_beta, n_closed, n_open, hf_ie_eV}`.
- `utils.py` — `HARTREE_TO_EV = 27.211386`, `ABS_TOL_EV = 3.0` eV, `_build_from_geometry()` (full molecule→SCF-inputs pipeline with basis caching).
- `templates.py` — test factory functions: `make_atom_scf_converged_test()`, `make_atom_koopmans_ie_test()`, `make_molecule_scf_converged_test()`, `make_dft_atom_scf_converged_test()`, etc. Eliminates duplication across `test_rhf.py`, `test_uhf.py`, `test_rohf.py`, `test_rks.py`, `test_uks.py`, `test_roks.py`.
- Module-scoped fixtures — each expensive SCF runs once, shared across 3 checks (converged, negative energy, IE match).

## Performance Notes
Bottleneck is `TwoElectronRepulsion` (4-center ERI). Optimizations applied:
- `lru_cache` on `double_factorial`, `normalization_constant`
- `boys_function_array` with downward recursion
- Bottom-up DP for `hermite_expansion_coefficients` and `hermite_coulomb_table`
- Schwarz screening to skip negligible ERI quartets
- Numba `@njit(cache=True)` on `_primitive_eri` and its math helpers (`_normalization_constant_jit`, `_hermite_expansion_coefficients_jit`, `_hermite_coulomb_table_jit`, `_boys_function_jit`) — JIT-compiled at first call, cached to `__pycache__`