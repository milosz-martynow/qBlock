"""Validation tests: Restricted Hartree-Fock Koopmans' theorem IE.

Verifies that the RHF implementation produces orbital energies consistent
with the ``hf_ie_eV`` reference values stored in ``validation_data.py``
via Koopmans' theorem:

    IE ≈ −ε_HOMO

Two groups of systems are tested:

* **Closed-shell atoms** (multiplicity == 1, from ``ATOMS_HOMO_ENERGIES``):
  each atom is placed at the origin and the HOMO is at index ``n_occ − 1``
  where ``n_occ = n_closed = n_electrons / 2``.

* **RHF molecules** (method == "RHF", from ``MOLECULES_HOMO_ENERGIES``):
  geometry from the experimental equilibrium structure; HOMO at
  ``n_closed − 1``.

Each system is run once (``scope="module"`` fixture) and three tests are
run against the shared result:

1. The SCF must have converged.
2. The total energy must be negative (bound system).
3. The Koopmans IE must match the ``hf_ie_eV`` reference within
   ``ABS_TOL_EV`` eV.  This tolerance accommodates the 3-21G / near-basis-
   set-limit difference while still detecting sign errors, unit errors, and
   incorrect orbital indexing.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from compute.solvers.wavefunction.hartree_fock.restricted_hartree_fock import (
    RestrictedHartreeFock,
)
from tests.validation.templates import (
    make_atom_koopmans_ie_test,
    make_atom_scf_converged_test,
    make_molecule_homo_ie_positive_test,
    make_molecule_koopmans_ie_test,
    make_molecule_scf_converged_test,
    make_molecule_total_energy_negative_test,
)
from tests.validation.utils import _build_from_geometry
from tests.validation.validation_data import (
    ATOMS_HOMO_ENERGIES,
    MOLECULES_HOMO_ENERGIES,
)

# ---------------------------------------------------------------------------
# Test data selection
# ---------------------------------------------------------------------------

# Closed-shell atoms: multiplicity == 1 → RHF
_rhf_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS_HOMO_ENERGIES.items()
    if "RHF" in entry["proposed_hartree_fock_approach"]
]

# RHF molecules
_rhf_mol_entries = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if "RHF" in entry["proposed_hartree_fock_approach"]
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rhf_atom_entries],
    ids=[key for key, _ in _rhf_atom_entries],
)
def rhf_atom_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], RestrictedHartreeFock]:
    """Run RHF SCF for one closed-shell atom (placed at origin) and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = RestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_electrons=entry["n_electrons"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rhf_mol_entries],
    ids=[key for key, _ in _rhf_mol_entries],
)
def rhf_molecule_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], RestrictedHartreeFock]:
    """Run RHF SCF for one closed-shell molecule and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = RestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_electrons=entry["n_electrons"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


# ---------------------------------------------------------------------------
# Tests — closed-shell atoms
# ---------------------------------------------------------------------------

# RHF-specific helper functions
_rhf_homo_idx: Callable[[Dict[str, Any]], int] = lambda entry: entry["n_closed"] - 1
_rhf_epsilon: Callable[[RestrictedHartreeFock], np.ndarray] = (
    lambda hf: hf.matrices["epsilon"]
)


def test_rhf_atom_scf_converged(rhf_atom_result) -> None:
    """RHF SCF converges for every closed-shell atom.

    Convergence is a prerequisite for all subsequent checks.  Failure here
    indicates either a stiff SCF for the given element or an implementation
    problem in the iteration loop.

    :param rhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_scf_converged_test("RHF")(rhf_atom_result)


def test_rhf_atom_koopmans_ie_vs_reference(rhf_atom_result) -> None:
    """Koopmans IE from the RHF HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The HOMO is the highest doubly-occupied orbital at index
    ``n_occ − 1 = n_closed − 1`` in the ascending-order eigenvalue array.

    :param rhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_koopmans_ie_test("RHF", _rhf_homo_idx, _rhf_epsilon)(
        rhf_atom_result
    )


# ---------------------------------------------------------------------------
# Tests — RHF molecules
# ---------------------------------------------------------------------------


def test_rhf_molecule_scf_converged(rhf_molecule_result) -> None:
    """RHF SCF converges for every closed-shell molecule.

    :param rhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_scf_converged_test("RHF")(rhf_molecule_result)


def test_rhf_molecule_total_energy_negative(rhf_molecule_result) -> None:
    """RHF total energy is negative for every closed-shell molecule.

    A positive total energy would indicate an unphysical result (molecule
    less stable than separated nuclei and electrons at infinity).

    :param rhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_total_energy_negative_test("RHF")(rhf_molecule_result)


def test_rhf_molecule_homo_ie_positive(rhf_molecule_result) -> None:
    """Koopmans IE from the RHF HOMO is positive (HOMO is a bound orbital).

    A negative IE would mean the HOMO energy is positive, i.e. the electron
    is not bound, which cannot be correct at the equilibrium geometry.

    :param rhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_homo_ie_positive_test("RHF", _rhf_homo_idx, _rhf_epsilon)(
        rhf_molecule_result
    )


def test_rhf_molecule_koopmans_ie_vs_reference(rhf_molecule_result) -> None:
    """Koopmans IE from the RHF HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The HOMO is the highest doubly-occupied orbital at index
    ``n_closed − 1`` in the ascending-order eigenvalue array.

    :param rhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_koopmans_ie_test("RHF", _rhf_homo_idx, _rhf_epsilon)(
        rhf_molecule_result
    )
