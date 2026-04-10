"""Validation tests: Unrestricted Hartree-Fock Koopmans' theorem IE.

Verifies that the UHF implementation produces alpha-spin orbital energies
consistent with the ``hf_ie_eV`` reference values stored in
``validation_data.py`` via Koopmans' theorem:

    IE ≈ −ε_α_HOMO

Two groups of systems are tested:

* **Open-shell atoms** (multiplicity > 1, from ``ATOMS_HOMO_ENERGIES``):
  each atom is placed at the origin.  The alpha-spin eigenvalues are stored
  in ascending order; the alpha HOMO is at index ``n_alpha − 1``.

* **UHF molecules** (method == "UHF", from ``MOLECULES_HOMO_ENERGIES``):
  geometry from the experimental equilibrium structure; alpha HOMO at
  ``n_alpha − 1``.

Because UHF allows independent optimisation of alpha and beta orbitals,
the Koopmans IE corresponds to the alpha-spin HOMO energy, which differs
slightly from the ROHF effective-Fock HOMO due to spin polarisation.  UHF
also introduces spin contamination (⟨S²⟩ ≠ S(S+1)); the tests do not
check the spin-purity but focus on the IE and convergence.

Each system is run once (``scope="module"`` fixture) and four tests are
applied to the shared result:

1. The SCF must have converged.
2. The total energy must be negative (bound system).
3. The alpha Koopmans IE must be positive (bound electron).
4. The Koopmans IE must match the ``hf_ie_eV`` reference within
   ``ABS_TOL_EV`` eV.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from q_block.solvers.wavefunction.hartree_fock.unrestricted_hartree_fock import (
    UnrestrictedHartreeFock,
)
from tests.validation_tests.templates import (
    make_atom_koopmans_ie_test,
    make_atom_scf_converged_test,
    make_molecule_homo_ie_positive_test,
    make_molecule_koopmans_ie_test,
    make_molecule_scf_converged_test,
    make_molecule_total_energy_negative_test,
)
from tests.validation_tests.utils import _build_from_geometry
from tests.validation_tests.validation_data import (
    ATOMS_HOMO_ENERGIES,
    MOLECULES_HOMO_ENERGIES,
)

# ---------------------------------------------------------------------------
# Test data selection
# ---------------------------------------------------------------------------

# Open-shell atoms: multiplicity > 1 → UHF
_uhf_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS_HOMO_ENERGIES.items()
    if "UHF" in entry["proposed_hartree_fock_approach"]
]

# UHF molecules
_uhf_mol_entries = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if "UHF" in entry["proposed_hartree_fock_approach"]
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _uhf_atom_entries],
    ids=[key for key, _ in _uhf_atom_entries],
)
def uhf_atom_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], UnrestrictedHartreeFock]:
    """Run UHF SCF for one open-shell atom (placed at origin) and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _uhf_mol_entries],
    ids=[key for key, _ in _uhf_mol_entries],
)
def uhf_molecule_result(
    request: pytest.FixtureRequest,
) -> Tuple[Dict[str, Any], UnrestrictedHartreeFock]:
    """Run UHF SCF for one open-shell molecule and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    hf = UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        max_iterations=entry["max_iterations"],
    ).run()
    return entry, hf


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------

# UHF-specific helper functions
_uhf_homo_idx: Callable[[Dict[str, Any]], int] = lambda entry: entry["n_alpha"] - 1
_uhf_epsilon: Callable[[UnrestrictedHartreeFock], np.ndarray] = (
    lambda hf: hf.matrices["epsilon_alpha"]
)


def test_uhf_atom_scf_converged(uhf_atom_result) -> None:
    """UHF SCF converges for every open-shell atom.

    :param uhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_scf_converged_test("UHF")(uhf_atom_result)


def test_uhf_atom_koopmans_ie_vs_reference(uhf_atom_result) -> None:
    """Koopmans IE from the UHF alpha HOMO agrees with ``hf_ie_eV``.

    The alpha HOMO is at index ``n_alpha − 1`` in the ascending-order
    alpha-spin eigenvalue array ``epsilon_alpha``.

    :param uhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_atom_koopmans_ie_test("UHF", _uhf_homo_idx, _uhf_epsilon)(
        uhf_atom_result
    )


# ---------------------------------------------------------------------------
# Tests — UHF molecules
# ---------------------------------------------------------------------------


def test_uhf_molecule_scf_converged(uhf_molecule_result) -> None:
    """UHF SCF converges for every open-shell molecule.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_scf_converged_test("UHF")(uhf_molecule_result)


def test_uhf_molecule_total_energy_negative(uhf_molecule_result) -> None:
    """UHF total energy is negative for every open-shell molecule.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_total_energy_negative_test("UHF")(uhf_molecule_result)


def test_uhf_molecule_alpha_homo_ie_positive(uhf_molecule_result) -> None:
    """Koopmans IE from the UHF alpha HOMO is positive (HOMO is a bound orbital).

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_homo_ie_positive_test("UHF", _uhf_homo_idx, _uhf_epsilon)(
        uhf_molecule_result
    )


def test_uhf_molecule_koopmans_ie_vs_reference(uhf_molecule_result) -> None:
    """Koopmans IE from the UHF alpha HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The alpha HOMO is at index ``n_alpha − 1`` in the ascending-order
    ``epsilon_alpha`` array.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    return make_molecule_koopmans_ie_test("UHF", _uhf_homo_idx, _uhf_epsilon)(
        uhf_molecule_result
    )
