"""Validation tests: Restricted Kohn-Sham (RKS) DFT.

Verifies that the RKS implementation produces physically reasonable
results for closed-shell systems with three XC functionals:

* **SVWN** (LDA) – Local Density Approximation
* **PBE**  (GGA) – Perdew–Burke–Ernzerhof
* **B3LYP** (hybrid) – Becke 3-parameter Lee–Yang–Parr

Two groups of systems are tested:

* **Closed-shell atoms** (multiplicity == 1, from ``ATOMS_HOMO_ENERGIES``):
  each atom is placed at the origin and the HOMO is at index ``n_occ − 1``
  where ``n_occ = n_closed = n_electrons / 2``.

* **RKS molecules** (closed-shell, from ``MOLECULES_HOMO_ENERGIES``):
  geometry from the experimental equilibrium structure; HOMO at
  ``n_closed − 1``.

Each system is run once per functional (``scope="module"`` fixture) and
four tests are applied:

1. The SCF must have converged.
2. The total energy must be negative (bound system).
3. The total energy must be finite (no NaN or Inf).
4. The Koopmans IE from the HOMO must be positive (bound electron).
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest

from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
    ExchangeCorrelationFunctional,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
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

# Closed-shell atoms: proposed_approach contains "RKS"
_rks_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS.items()
    if "RKS" in entry["proposed_approach"]
]

# RKS molecules: proposed_approach contains "RKS"
_rks_mol_entries: List[Tuple[str, Dict[str, Any]]] = [
    (key, entry)
    for key, entry in MOLECULES.items()
    if "RKS" in entry["proposed_approach"]
]

# Functionals to test
_functionals: List[Tuple[str, ExchangeCorrelationFunctional]] = [
    ("SVWN", SVWN()),
    ("PBE", PBE()),
    ("B3LYP", B3LYP()),
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[
        (atom_key, atom_entry, func_name, func)
        for atom_key, atom_entry in _rks_atom_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{atom_key}_{func_name}"
        for atom_key, _ in _rks_atom_entries
        for func_name, _ in _functionals
    ],
)
def rks_atom_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, RestrictedKohnSham]:
    """Run RKS SCF for one closed-shell atom with one functional.

    :returns: ``(entry, functional_name, rks)``
    :rtype: Tuple[Dict[str, Any], str, RestrictedKohnSham]
    """
    atom_key, entry, func_name, func = request.param
    cgtos = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=func,
        n_electrons=entry["n_electrons"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
    ).run()
    return entry, func_name, rks


@pytest.fixture(
    scope="module",
    params=[
        (mol_key, mol_entry, func_name, func)
        for mol_key, mol_entry in _rks_mol_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{mol_key}_{func_name}"
        for mol_key, _ in _rks_mol_entries
        for func_name, _ in _functionals
    ],
)
def rks_molecule_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, RestrictedKohnSham]:
    """Run RKS SCF for one closed-shell molecule with one functional.

    :returns: ``(entry, functional_name, rks)``
    :rtype: Tuple[Dict[str, Any], str, RestrictedKohnSham]
    """
    mol_key, entry, func_name, func = request.param
    cgtos = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        functional=func,
        n_electrons=entry["n_electrons"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
    ).run()
    return entry, func_name, rks


# ---------------------------------------------------------------------------
# RKS-specific helper functions
# ---------------------------------------------------------------------------
_rks_homo_idx: Callable[[Dict[str, Any]], int] = lambda entry: entry["n_closed"] - 1
_rks_epsilon: Callable[[RestrictedKohnSham], np.ndarray] = (
    lambda ks: ks.matrices["epsilon"]
)


# ---------------------------------------------------------------------------
# Tests — closed-shell atoms
# ---------------------------------------------------------------------------


def test_rks_atom_scf_converged(rks_atom_result) -> None:
    """RKS SCF converges for every closed-shell atom.

    Convergence is a prerequisite for all subsequent checks.  Failure here
    indicates either a stiff SCF for the given element or an implementation
    problem in the iteration loop.

    :param rks_atom_result: Pytest fixture providing ``(entry, func_name, rks)``.
    """
    return make_dft_atom_scf_converged_test("RKS")(rks_atom_result)


def test_rks_atom_total_energy_negative(rks_atom_result) -> None:
    """RKS total energy is negative for every closed-shell atom.

    A positive total energy would indicate an unphysical result.

    :param rks_atom_result: Pytest fixture providing ``(entry, func_name, rks)``.
    """
    return make_dft_atom_total_energy_negative_test("RKS")(rks_atom_result)


def test_rks_atom_total_energy_finite(rks_atom_result) -> None:
    """RKS total energy is finite (no NaN or Inf) for every closed-shell atom.

    :param rks_atom_result: Pytest fixture providing ``(entry, func_name, rks)``.
    """
    return make_dft_atom_total_energy_finite_test("RKS")(rks_atom_result)


def test_rks_atom_homo_ie_positive(rks_atom_result) -> None:
    """Koopmans IE from the RKS HOMO is positive (bound electron).

    :param rks_atom_result: Pytest fixture providing ``(entry, func_name, rks)``.
    """
    return make_dft_atom_homo_ie_positive_test(
        "RKS", _rks_homo_idx, _rks_epsilon
    )(rks_atom_result)


def test_rks_atom_koopmans_ie_vs_experiment(rks_atom_result) -> None:
    """Koopmans IE from the RKS HOMO agrees with ``energies['experiment'][0]['value']``.

    Uses Janak's theorem (IE ≈ −ε_HOMO) to compare the DFT HOMO
    eigenvalue against the experimental ionisation energy.

    :param rks_atom_result: Pytest fixture providing ``(entry, func_name, rks)``.
    """
    return make_dft_atom_koopmans_ie_test(
        "RKS", _rks_homo_idx, _rks_epsilon
    )(rks_atom_result)


# ---------------------------------------------------------------------------
# Tests — RKS molecules
# ---------------------------------------------------------------------------


def test_rks_molecule_scf_converged(rks_molecule_result) -> None:
    """RKS SCF converges for every closed-shell molecule.

    :param rks_molecule_result: Pytest fixture providing
        ``(entry, func_name, rks)``.
    """
    return make_dft_molecule_scf_converged_test("RKS")(rks_molecule_result)


def test_rks_molecule_total_energy_negative(rks_molecule_result) -> None:
    """RKS total energy is negative for every closed-shell molecule.

    A positive total energy would indicate an unphysical result (molecule
    less stable than separated nuclei and electrons at infinity).

    :param rks_molecule_result: Pytest fixture providing
        ``(entry, func_name, rks)``.
    """
    return make_dft_molecule_total_energy_negative_test("RKS")(rks_molecule_result)


def test_rks_molecule_total_energy_finite(rks_molecule_result) -> None:
    """RKS total energy is finite (no NaN or Inf).

    :param rks_molecule_result: Pytest fixture providing
        ``(entry, func_name, rks)``.
    """
    return make_dft_molecule_total_energy_finite_test("RKS")(rks_molecule_result)


def test_rks_molecule_homo_ie_positive(rks_molecule_result) -> None:
    """Koopmans IE from the RKS HOMO is positive (bound electron).

    A negative IE would mean the HOMO energy is positive, i.e. the electron
    is not bound, which cannot be correct at the equilibrium geometry.

    :param rks_molecule_result: Pytest fixture providing
        ``(entry, func_name, rks)``.
    """
    return make_dft_molecule_homo_ie_positive_test(
        "RKS", _rks_homo_idx, _rks_epsilon
    )(rks_molecule_result)


def test_rks_molecule_koopmans_ie_vs_experiment(rks_molecule_result) -> None:
    """Koopmans IE from the RKS HOMO agrees with ``energies['experiment'][0]['value']``.

    Uses Janak's theorem (IE ≈ −ε_HOMO) to compare the DFT HOMO
    eigenvalue against the experimental ionisation energy.

    :param rks_molecule_result: Pytest fixture providing
        ``(entry, func_name, rks)``.
    """
    return make_dft_molecule_koopmans_ie_test(
        "RKS", _rks_homo_idx, _rks_epsilon
    )(rks_molecule_result)
