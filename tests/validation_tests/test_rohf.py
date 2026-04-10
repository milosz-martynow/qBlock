"""Validation tests: Restricted Open-Shell Hartree-Fock Koopmans' theorem IE.

Verifies that the ROHF implementation produces orbital energies consistent
with the ``hf_ie_eV`` reference values stored in ``validation_data.py``
via Koopmans' theorem:

    IE ≈ −ε_HOMO

Two groups of systems are tested:

* **Open-shell atoms** (multiplicity > 1, from ``ATOMS_HOMO_ENERGIES``):
  each atom is placed at the origin.  The effective-Fock eigenvalues are
  stored in ascending order; the HOMO is the last singly-occupied orbital
  at index ``n_closed + n_open − 1``.

* **ROHF molecules** (method == "ROHF", from ``MOLECULES_HOMO_ENERGIES``):
  geometry from the experimental equilibrium structure; HOMO at
  ``n_closed + n_open − 1``.

Each system is run once (``scope="module"`` fixture) and three tests are
applied to the shared result:

1. The SCF must have converged.
2. The total energy must be negative (bound system).
3. The Koopmans IE must match the ``hf_ie_eV`` reference within
   ``ABS_TOL_EV`` eV.
"""

import pytest

import numpy as np

from typing import Any, Callable, Dict, Tuple

from q_block.solvers.wavefunction.hartree_fock.restricted_open_shell_hartree_fock import (
    RestrictedOpenShellHartreeFock,
)
from tests.validation_tests.utils import _build_from_geometry
from tests.validation_tests.validation_data import (
    ATOMS_HOMO_ENERGIES,
    MOLECULES_HOMO_ENERGIES,
)
from tests.validation_tests.templates import (
    make_atom_scf_converged_test,
    make_atom_koopmans_ie_test,
    make_molecule_scf_converged_test,
    make_molecule_total_energy_negative_test,
    make_molecule_homo_ie_positive_test,
    make_molecule_koopmans_ie_test,
)


# ---------------------------------------------------------------------------
# Test data selection
# ---------------------------------------------------------------------------

# Open-shell atoms: multiplicity > 1 → ROHF
_rohf_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS_HOMO_ENERGIES.items()
    if "ROHF" in entry["proposed_hartree_fock_approach"]
]

# ROHF molecules
_rohf_mol_entries = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if "ROHF" in entry["proposed_hartree_fock_approach"]
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rohf_atom_entries],
    ids=[key for key, _ in _rohf_atom_entries],
)
def rohf_atom_result(request: pytest.FixtureRequest) -> Tuple[Dict[str, Any], RestrictedOpenShellHartreeFock]:
    """Run ROHF SCF for one open-shell atom (placed at origin) and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rohf_mol_entries],
    ids=[key for key, _ in _rohf_mol_entries],
)
def rohf_molecule_result(request: pytest.FixtureRequest) -> Tuple[Dict[str, Any], RestrictedOpenShellHartreeFock]:
    """Run ROHF SCF for one open-shell molecule and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------

# ROHF-specific helper functions
_rohf_homo_idx: Callable[[Dict[str, Any]], int] = lambda entry: entry["n_closed"] + entry["n_open"] - 1
_rohf_epsilon: Callable[[RestrictedOpenShellHartreeFock], np.ndarray] = lambda hf: hf.matrices["epsilon"]


def test_rohf_atom_scf_converged(rohf_atom_result) -> None:
    """ROHF SCF converges for every open-shell atom.

    :param rohf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_scf_converged_test("ROHF")(rohf_atom_result)


def test_rohf_atom_koopmans_ie_vs_reference(rohf_atom_result) -> None:
    """Koopmans IE from the ROHF HOMO (last SOMO) agrees with ``hf_ie_eV``.

    The HOMO is the highest singly-occupied orbital at index
    ``n_closed + n_open − 1`` in the ascending-order effective-Fock
    eigenvalue array.

    :param rohf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_koopmans_ie_test("ROHF", _rohf_homo_idx, _rohf_epsilon)(rohf_atom_result)


# ---------------------------------------------------------------------------
# Tests — ROHF molecules
# ---------------------------------------------------------------------------


def test_rohf_molecule_scf_converged(rohf_molecule_result) -> None:
    """ROHF SCF converges for every open-shell molecule.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_scf_converged_test("ROHF")(rohf_molecule_result)


def test_rohf_molecule_total_energy_negative(rohf_molecule_result) -> None:
    """ROHF total energy is negative for every open-shell molecule.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_total_energy_negative_test("ROHF")(rohf_molecule_result)


def test_rohf_molecule_homo_ie_positive(rohf_molecule_result) -> None:
    """Koopmans IE from the ROHF HOMO is positive (HOMO is a bound orbital).

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_homo_ie_positive_test("ROHF", _rohf_homo_idx, _rohf_epsilon)(rohf_molecule_result)


def test_rohf_molecule_koopmans_ie_vs_reference(rohf_molecule_result) -> None:
    """Koopmans IE from the ROHF HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The HOMO is the last singly-occupied orbital at index
    ``n_closed + n_open − 1`` in the ascending-order effective-Fock
    eigenvalue array.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_koopmans_ie_test("ROHF", _rohf_homo_idx, _rohf_epsilon)(rohf_molecule_result)
