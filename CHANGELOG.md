# Changelog

All notable and major changes to this project will be documented in this file.

## [2] - 2026-05-15

### Added
- Basis set as a piece of package. now it is possible to call them via e.g. "from q_block.compute.environment.constants.numerical.pople import G631"

### Changed
- ...

### Fixed
- SCF variables are inherited by KS and HF methods.
- All validation tests are passing - this includes e.g., eliminating physical vs numerical constrains examples (some of examples can't run  for example for DFT on PBE due to physical problems that PBE does not cover etc.)

## [1.0.7] - 2026-05-05

### Added
- ROKS - Restricted Open Shell Kohn Sham DFT `q_block\compute\solvers\electronic_density\kohn_sham\restricted_open_shell_kohn_sham.py` with tests.
- Sphinx documentation  intialization
- Add automatic calculation of nuclear energy calculation instead of defining it in the SCF instances.
### Changed
- ...

### Fixed
- ...
