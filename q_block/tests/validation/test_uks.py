"""Validation tests: Unrestricted Kohn-Sham (UKS) DFT.

Verifies that the UKS implementation produces physically reasonable
results for open-shell systems with three XC functionals:

* **SVWN** (LDA) – Local Density Approximation
* **PBE**  (GGA) – Perdew–Burke–Ernzerhof
* **B3LYP** (hybrid) – Becke 3-parameter Lee–Yang–Parr

Two groups of systems are tested:

* **Open-shell atoms** (multiplicity > 1, from ``ATOMS_HOMO_ENERGIES``):
  each atom is placed at the origin.  The alpha-spin eigenvalues are stored
  in ascending order; the alpha HOMO is at index ``n_alpha − 1``.

* **UKS molecules** (open-shell, from ``MOLECULES_HOMO_ENERGIES``):
  geometry from the experimental equilibrium structure; alpha HOMO at
  ``n_alpha − 1``.

Each system is run once per functional (``scope="module"`` fixture) and
four tests are applied:

1. The SCF must have converged.
2. The total energy must be negative (bound system).
3. The total energy must be finite (no NaN or Inf).
4. The Koopmans IE from the alpha HOMO must be positive (bound electron).
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from q_block.compute.models.basis_functions import ContractedGaussianTypeOrbital
from q_block.compute.models.integrals.two_electron_repulsion import TwoElectronRepulsion
from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
    ExchangeCorrelationFunctional,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    UnrestrictedKohnSham,
)
from q_block.tests.validation.templates import (
    make_dft_atom_homo_ie_positive_test,
    make_dft_atom_koopmans_ie_test,
    make_dft_atom_scf_converged_test,
    make_dft_atom_total_energy_finite_test,
    make_dft_atom_total_energy_negative_test,
    make_dft_molecule_homo_ie_positive_test,
    make_dft_molecule_koopmans_ie_test,
    make_dft_molecule_scf_converged_test,
    make_dft_molecule_total_energy_finite_test,
    make_dft_molecule_total_energy_negative_test,
)
from q_block.tests.validation.utils import _build_from_geometry
from q_block.tests.validation.validation_data import (
    ATOMS,
    MOLECULES,
)

# ---------------------------------------------------------------------------
# Test data selection
# ---------------------------------------------------------------------------

# Open-shell atoms: proposed_approach contains "UKS"
_uks_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS.items()
    if "UKS" in entry["proposed_approach"]
]

# UKS molecules: proposed_approach contains "UKS"
_uks_mol_entries: List[Tuple[str, Dict[str, Any]]] = [
    (key, entry)
    for key, entry in MOLECULES.items()
    if "UKS" in entry["proposed_approach"]
]

# Functionals to test
_functionals: List[Tuple[str, ExchangeCorrelationFunctional]] = [
    ("SVWN", SVWN()),
    ("PBE", PBE()),
    ("B3LYP", B3LYP()),
]

# ---------------------------------------------------------------------------
# Module-level caches: CGTOs and ERI built once per atom/molecule key.
# Each cache entry is (cgtos, eri_tensor) and is populated lazily on first
# access during fixture execution, then reused for subsequent functionals.
# ---------------------------------------------------------------------------
_uks_atom_cache: Dict[
    str, Tuple[List[ContractedGaussianTypeOrbital], np.ndarray]
] = {}
_uks_mol_cache: Dict[
    str, Tuple[List[ContractedGaussianTypeOrbital], np.ndarray]
] = {}

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        (atom_key, atom_entry, func_name, func)
        for atom_key, atom_entry in _uks_atom_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{atom_key}_{func_name}"
        for atom_key, _ in _uks_atom_entries
        for func_name, _ in _functionals
    ],
)
def uks_atom_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, UnrestrictedKohnSham]:
    """Run UKS SCF for one open-shell atom with one functional.

    CGTOs and the ERI tensor are built once per atom (cached in
    ``_uks_atom_cache``) and reused across all three functionals, avoiding
    redundant O(N\u2074) ERI recomputation.

    :returns: ``(entry, functional_name, uks)``
    :rtype: Tuple[Dict[str, Any], str, UnrestrictedKohnSham]
    """
    atom_key, entry, func_name, func = request.param
    if atom_key not in _uks_atom_cache:
        cgtos = _build_from_geometry(
            geometry=[
                {"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}
            ],
            multiplicity=entry["multiplicity"],
            basis_set_filename=entry["proposed_basis_set"],
        )
        eri = TwoElectronRepulsion(cgtos=cgtos).tensor
        _uks_atom_cache[atom_key] = (cgtos, eri)
    cgtos, eri = _uks_atom_cache[atom_key]
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=func,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
        precomputed_eri=eri,
    ).run()
    return entry, func_name, uks


@pytest.fixture(
    scope="module",
    params=[
        (mol_key, mol_entry, func_name, func)
        for mol_key, mol_entry in _uks_mol_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{mol_key}_{func_name}"
        for mol_key, _ in _uks_mol_entries
        for func_name, _ in _functionals
    ],
)
def uks_molecule_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, UnrestrictedKohnSham]:
    """Run UKS SCF for one open-shell molecule with one functional.

    CGTOs and the ERI tensor are built once per molecule (cached in
    ``_uks_mol_cache``) and reused across all three functionals, avoiding
    redundant O(N\u2074) ERI recomputation.

    :returns: ``(entry, functional_name, uks)``
    :rtype: Tuple[Dict[str, Any], str, UnrestrictedKohnSham]
    """
    mol_key, entry, func_name, func = request.param
    if mol_key not in _uks_mol_cache:
        cgtos = _build_from_geometry(
            geometry=entry["geometry"],
            multiplicity=entry["multiplicity"],
            basis_set_filename=entry["proposed_basis_set"],
        )
        eri = TwoElectronRepulsion(cgtos=cgtos).tensor
        _uks_mol_cache[mol_key] = (cgtos, eri)
    cgtos, eri = _uks_mol_cache[mol_key]
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        functional=func,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
        precomputed_eri=eri,
    ).run()
    return entry, func_name, uks


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# UKS-specific helper functions
# ---------------------------------------------------------------------------
_uks_homo_idx: Callable[[Dict[str, Any]], int] = lambda entry: entry["n_alpha"] - 1
_uks_epsilon: Callable[[UnrestrictedKohnSham], np.ndarray] = (
    lambda ks: ks.matrices["epsilon_alpha"]
)


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------


def test_uks_atom_scf_converged(uks_atom_result) -> None:
    """UKS SCF converges for every open-shell atom.

    Convergence is a prerequisite for all subsequent checks.  Failure here
    indicates either a stiff SCF for the given element or an implementation
    problem in the iteration loop.

    :param uks_atom_result: Pytest fixture providing ``(entry, func_name, uks)``.
    """
    return make_dft_atom_scf_converged_test("UKS")(uks_atom_result)


def test_uks_atom_total_energy_negative(uks_atom_result) -> None:
    """UKS total energy is negative for every open-shell atom.

    A positive total energy would indicate an unphysical result.

    :param uks_atom_result: Pytest fixture providing ``(entry, func_name, uks)``.
    """
    return make_dft_atom_total_energy_negative_test("UKS")(uks_atom_result)


def test_uks_atom_total_energy_finite(uks_atom_result) -> None:
    """UKS total energy is finite (no NaN or Inf) for every open-shell atom.

    :param uks_atom_result: Pytest fixture providing ``(entry, func_name, uks)``.
    """
    return make_dft_atom_total_energy_finite_test("UKS")(uks_atom_result)


def test_uks_atom_alpha_homo_ie_positive(uks_atom_result) -> None:
    """Koopmans IE from the UKS alpha HOMO is positive (bound electron).

    :param uks_atom_result: Pytest fixture providing ``(entry, func_name, uks)``.
    """
    return make_dft_atom_homo_ie_positive_test(
        "UKS", _uks_homo_idx, _uks_epsilon
    )(uks_atom_result)


def test_uks_atom_koopmans_ie_vs_experiment(uks_atom_result) -> None:
    """Koopmans IE from the UKS alpha HOMO agrees with ``energies['experiment'][0]['value']``.

    Uses Janak's theorem (IE ≈ −ε_HOMO) to compare the DFT alpha HOMO
    eigenvalue against the experimental ionisation energy.

    :param uks_atom_result: Pytest fixture providing ``(entry, func_name, uks)``.
    """
    return make_dft_atom_koopmans_ie_test(
        "UKS", _uks_homo_idx, _uks_epsilon
    )(uks_atom_result)


# ---------------------------------------------------------------------------
# Tests — UKS molecules
# ---------------------------------------------------------------------------


def test_uks_molecule_scf_converged(uks_molecule_result) -> None:
    """UKS SCF converges for every open-shell molecule.

    :param uks_molecule_result: Pytest fixture providing
        ``(entry, func_name, uks)``.
    """
    return make_dft_molecule_scf_converged_test("UKS")(uks_molecule_result)


def test_uks_molecule_total_energy_negative(uks_molecule_result) -> None:
    """UKS total energy is negative for every open-shell molecule.

    A positive total energy would indicate an unphysical result (molecule
    less stable than separated nuclei and electrons at infinity).

    :param uks_molecule_result: Pytest fixture providing
        ``(entry, func_name, uks)``.
    """
    return make_dft_molecule_total_energy_negative_test("UKS")(uks_molecule_result)


def test_uks_molecule_total_energy_finite(uks_molecule_result) -> None:
    """UKS total energy is finite (no NaN or Inf).

    :param uks_molecule_result: Pytest fixture providing
        ``(entry, func_name, uks)``.
    """
    return make_dft_molecule_total_energy_finite_test("UKS")(uks_molecule_result)


def test_uks_molecule_alpha_homo_ie_positive(uks_molecule_result) -> None:
    """Koopmans IE from the UKS alpha HOMO is positive (bound electron).

    A negative IE would mean the alpha HOMO energy is positive, i.e. the
    electron is not bound, which cannot be correct at the equilibrium
    geometry.

    :param uks_molecule_result: Pytest fixture providing
        ``(entry, func_name, uks)``.
    """
    return make_dft_molecule_homo_ie_positive_test(
        "UKS", _uks_homo_idx, _uks_epsilon
    )(uks_molecule_result)


def test_uks_molecule_koopmans_ie_vs_experiment(uks_molecule_result) -> None:
    """Koopmans IE from the UKS alpha HOMO agrees with ``energies['experiment'][0]['value']``.

    Uses Janak's theorem (IE ≈ −ε_HOMO) to compare the DFT alpha HOMO
    eigenvalue against the experimental ionisation energy.

    :param uks_molecule_result: Pytest fixture providing
        ``(entry, func_name, uks)``.
    """
    return make_dft_molecule_koopmans_ie_test(
        "UKS", _uks_homo_idx, _uks_epsilon
    )(uks_molecule_result)
