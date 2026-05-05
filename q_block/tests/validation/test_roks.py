"""Validation tests: Restricted Open-Shell Kohn-Sham (ROKS) DFT.

Verifies that the ROKS implementation produces physically reasonable
results for open-shell systems with three XC functionals:

* **SVWN** (LDA) – Local Density Approximation
* **PBE**  (GGA) – Perdew–Burke–Ernzerhof
* **B3LYP** (hybrid) – Becke 3-parameter Lee–Yang–Parr

Two groups of systems are tested:

* **Open-shell atoms** (``"ROKS" in proposed_approach``, from ``ATOMS``):
  each atom is placed at the origin.  The effective-Fock eigenvalues are
  stored in ascending order; the HOMO is the last occupied orbital at
  index ``n_closed + n_open - 1``.

* **ROKS molecules** (``"ROKS" in proposed_approach``, from ``MOLECULES``):
  geometry from the experimental equilibrium structure; HOMO at
  ``n_closed + n_open - 1``.

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
    RestrictedOpenShellKohnSham,
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

_roks_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS.items()
    if "ROKS" in entry["proposed_approach"]
]

_roks_mol_entries: List[Tuple[str, Dict[str, Any]]] = [
    (key, entry)
    for key, entry in MOLECULES.items()
    if "ROKS" in entry["proposed_approach"]
]

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
        for atom_key, atom_entry in _roks_atom_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{atom_key}_{func_name}"
        for atom_key, _ in _roks_atom_entries
        for func_name, _ in _functionals
    ],
)
def roks_atom_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, RestrictedOpenShellKohnSham]:
    """Run ROKS SCF for one open-shell atom with one functional.

    :returns: ``(entry, functional_name, roks)``
    :rtype: Tuple[Dict[str, Any], str, RestrictedOpenShellKohnSham]
    """
    atom_key, entry, func_name, func = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        functional=func,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
    ).run()
    return entry, func_name, roks


@pytest.fixture(
    scope="module",
    params=[
        (mol_key, mol_entry, func_name, func)
        for mol_key, mol_entry in _roks_mol_entries
        for func_name, func in _functionals
    ],
    ids=[
        f"{mol_key}_{func_name}"
        for mol_key, _ in _roks_mol_entries
        for func_name, _ in _functionals
    ],
)
def roks_molecule_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], str, RestrictedOpenShellKohnSham]:
    """Run ROKS SCF for one open-shell molecule with one functional.

    :returns: ``(entry, functional_name, roks)``
    :rtype: Tuple[Dict[str, Any], str, RestrictedOpenShellKohnSham]
    """
    mol_key, entry, func_name, func = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    roks = RestrictedOpenShellKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        functional=func,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
    ).run()
    return entry, func_name, roks


# ---------------------------------------------------------------------------
# ROKS-specific helper functions
# ---------------------------------------------------------------------------
_roks_homo_idx: Callable[[Dict[str, Any]], int] = (
    lambda entry: entry["n_closed"] + entry["n_open"] - 1
)
_roks_epsilon: Callable[[RestrictedOpenShellKohnSham], np.ndarray] = (
    lambda ks: ks.matrices["epsilon"]
)


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------


def test_roks_atom_scf_converged(roks_atom_result) -> None:
    """ROKS SCF converges for every open-shell atom.

    Convergence is a prerequisite for all subsequent checks.  Failure here
    indicates either a stiff SCF for the given element or an implementation
    problem in the iteration loop.

    :param roks_atom_result: Pytest fixture providing ``(entry, func_name, roks)``.
    """
    return make_dft_atom_scf_converged_test("ROKS")(roks_atom_result)


def test_roks_atom_total_energy_negative(roks_atom_result) -> None:
    """ROKS total energy is negative for every open-shell atom.

    A positive total energy would indicate an unphysical result.

    :param roks_atom_result: Pytest fixture providing ``(entry, func_name, roks)``.
    """
    return make_dft_atom_total_energy_negative_test("ROKS")(roks_atom_result)


def test_roks_atom_total_energy_finite(roks_atom_result) -> None:
    """ROKS total energy is finite (no NaN or Inf) for every open-shell atom.

    :param roks_atom_result: Pytest fixture providing ``(entry, func_name, roks)``.
    """
    return make_dft_atom_total_energy_finite_test("ROKS")(roks_atom_result)


def test_roks_atom_homo_ie_positive(roks_atom_result) -> None:
    """Koopmans IE from the ROKS HOMO is positive (bound electron).

    Uses the effective-Fock HOMO eigenvalue.  A negative IE would indicate
    an unbound orbital, which cannot be correct at equilibrium.

    :param roks_atom_result: Pytest fixture providing ``(entry, func_name, roks)``.
    """
    return make_dft_atom_homo_ie_positive_test(
        "ROKS", _roks_homo_idx, _roks_epsilon
    )(roks_atom_result)


def test_roks_atom_koopmans_ie_vs_experiment(roks_atom_result) -> None:
    """Koopmans IE from the ROKS HOMO agrees with ``energies['experiment'][0]['value']``.

    Uses Janak's theorem (IE ≈ −ε_HOMO) to compare the effective-Fock
    HOMO eigenvalue against the experimental ionisation energy.

    :param roks_atom_result: Pytest fixture providing ``(entry, func_name, roks)``.
    """
    return make_dft_atom_koopmans_ie_test(
        "ROKS", _roks_homo_idx, _roks_epsilon
    )(roks_atom_result)


# ---------------------------------------------------------------------------
# Tests — ROKS molecules
# ---------------------------------------------------------------------------


def test_roks_molecule_scf_converged(roks_molecule_result) -> None:
    """ROKS SCF converges for every open-shell molecule.

    :param roks_molecule_result: Pytest fixture providing
        ``(entry, func_name, roks)``.
    """
    return make_dft_molecule_scf_converged_test("ROKS")(roks_molecule_result)


def test_roks_molecule_total_energy_negative(roks_molecule_result) -> None:
    """ROKS total energy is negative for every open-shell molecule.

    :param roks_molecule_result: Pytest fixture providing
        ``(entry, func_name, roks)``.
    """
    return make_dft_molecule_total_energy_negative_test("ROKS")(roks_molecule_result)


def test_roks_molecule_total_energy_finite(roks_molecule_result) -> None:
    """ROKS total energy is finite (no NaN or Inf).

    :param roks_molecule_result: Pytest fixture providing
        ``(entry, func_name, roks)``.
    """
    return make_dft_molecule_total_energy_finite_test("ROKS")(roks_molecule_result)


def test_roks_molecule_homo_ie_positive(roks_molecule_result) -> None:
    """Koopmans IE from the ROKS HOMO is positive (bound electron).

    :param roks_molecule_result: Pytest fixture providing
        ``(entry, func_name, roks)``.
    """
    return make_dft_molecule_homo_ie_positive_test(
        "ROKS", _roks_homo_idx, _roks_epsilon
    )(roks_molecule_result)


def test_roks_molecule_koopmans_ie_vs_experiment(roks_molecule_result) -> None:
    """Koopmans IE from the ROKS HOMO agrees with ``energies['experiment'][0]['value']``.

    :param roks_molecule_result: Pytest fixture providing
        ``(entry, func_name, roks)``.
    """
    return make_dft_molecule_koopmans_ie_test(
        "ROKS", _roks_homo_idx, _roks_epsilon
    )(roks_molecule_result)
