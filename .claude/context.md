# Project Context

**qBlock** — Electronic structure library in Python 3.12. Author: Miłosz Martynow. Implements Hartree-Fock (RHF, UHF, ROHF) via McMurchie-Davidson integrals and SCF with DIIS acceleration.

## Package Layers
- `constants/` Pure data, no logic beyond dict inversion.
- `io/` Input parsing and data wrangling.
- `methods/` Electronic structure algorithms: SCF loop, DIIS, HF variants (RHF, UHF, ROHF).
- `models/` Quantum-structure models of atoms and electrons.
- `systems/` Atomic systems containers: atoms, molecules, crystals.
- `theory/` Physics: basis functions, integrals, nuclear repulsion.

## Root Files
- `setup.py` — Package definition.
- `README.md` — Setup instructions and project description.

## Other Directories
- `data/basis_set/pople/` — Gaussian basis set files like: `STO-3G`, `3-21G`.
- `examples/` — runnable scripts demonstrating the API (molecule creation, integrals, SCF, GTO dataframe).
- `architecture/` — PlantUML diagrams: `scf_activity_diagram.puml`, `use_case_diagram.puml`, block definition diagrams for each sub-package (`io`, `methods`, `models`, `systems`, `theory`).

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
black --config=.blackrc .\q_block\ .\tests\ setup.py
pylint --rcfile=.pylintrc .\q_block\ .\tests\ setup.py
pytest .
```

## Test Suites

### Unit Tests (`tests/unit_tests/`)
Per-module, parametrized, no test classes — bare functions with `@pytest.mark.parametrize`.
- `constants.py` — shared fixtures: pre-loaded basis sets (`BASIS_3_21G`, `BASIS_STO_3G`, …), path constants (`BASIS_ROOT`, `GOLDEN_ROOT`, `GEOMETRIES_DIR`), `ORIGIN` coordinate.
- `utils.py` — shared helpers: `ALL_ORBITAL_COMPONENTS`, `ORBITAL_LABELS`, `orbital_id()`, `get_orbital_ids()`.
- `verification_data/` — golden reference files: `expected_atom_empirical.py`, `expected_atom_pure.py`, `gto_population/*.json`, `geometries/*.xyz` (H2, water, azobenzene, tetraethylammonium), `tools/` (generator scripts).
- Mirror structure: `tests/unit_tests/io/test_coordinates.py` ↔ `q_block/io/coordinates.py`, etc.

### Validation Tests (`tests/validation_tests/`)
End-to-end HF calculations verifying Koopmans' theorem ionization energies.
- `validation_data.py` — reference dicts (`ATOMS_HOMO_ENERGIES`, `MOLECULES_HOMO_ENERGIES`) mapping Z (1–54) → `{symbol, config, multiplicity, n_electrons, n_alpha, n_beta, n_closed, n_open, hf_ie_eV}`.
- `utils.py` — `HARTREE_TO_EV = 27.211386`, `ABS_TOL_EV = 3.0` eV, `_build_from_geometry()` (full molecule→SCF-inputs pipeline with basis caching).
- `templates.py` — test factory functions: `make_atom_scf_converged_test()`, `make_atom_koopmans_ie_test()`, `make_molecule_scf_converged_test()`, etc. Eliminates duplication across `test_rhf.py`, `test_uhf.py`, `test_rohf.py`.
- Module-scoped fixtures — each expensive SCF runs once, shared across 3 checks (converged, negative energy, IE match).

## Performance Notes
Bottleneck is `TwoElectronRepulsion` (4-center ERI). Optimizations applied:
- `lru_cache` on `double_factorial`, `normalization_constant`
- `boys_function_array` with downward recursion
- Bottom-up DP for `hermite_expansion_coefficients` and `hermite_coulomb_table`
- Schwarz screening to skip negligible ERI quartets