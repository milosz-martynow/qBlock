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

from q_block.methods.wavefunction.hartree_fock.restricted_open_shell_hartree_fock import (
    RestrictedOpenShellHartreeFock,
)
from tests.validation_tests.utils import ABS_TOL_EV, HARTREE_TO_EV, _build_from_geometry
from tests.validation_tests.validation_data import (
    ATOMS_HOMO_ENERGIES,
    MOLECULES_HOMO_ENERGIES,
)


# ---------------------------------------------------------------------------
# Test data selection
# ---------------------------------------------------------------------------

# Open-shell atoms: multiplicity > 1 → ROHF
_rohf_atom_entries = [
    (f"Z{z}_{entry['symbol']}", entry)
    for z, entry in ATOMS_HOMO_ENERGIES.items()
    if entry["multiplicity"] > 1
]

# ROHF molecules
_rohf_mol_entries = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if entry["method"] == "ROHF"
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rohf_atom_entries],
    ids=[key for key, _ in _rohf_atom_entries],
)
def rohf_atom_result(request):
    """Run ROHF SCF for one open-shell atom (placed at origin) and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
    )
    hf = RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        max_iterations=200,
    ).run()
    return entry, hf


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _rohf_mol_entries],
    ids=[key for key, _ in _rohf_mol_entries],
)
def rohf_molecule_result(request):
    """Run ROHF SCF for one open-shell molecule and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
    )
    hf = RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_closed=entry["n_closed"],
        n_open=entry["n_open"],
        max_iterations=200,
    ).run()
    return entry, hf


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------


def test_rohf_atom_scf_converged(rohf_atom_result) -> None:
    """ROHF SCF converges for every open-shell atom.

    :param rohf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_atom_result
    assert hf.converged, (
        f"ROHF SCF did not converge for atom {entry['symbol']} "
        f"(Z={entry['n_electrons']})."
    )


def test_rohf_atom_koopmans_ie_vs_reference(rohf_atom_result) -> None:
    """Koopmans IE from the ROHF HOMO (last SOMO) agrees with ``hf_ie_eV``.

    The HOMO is the highest singly-occupied orbital at index
    ``n_closed + n_open − 1`` in the ascending-order effective-Fock
    eigenvalue array.

    :param rohf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_atom_result
    homo_idx = entry["n_closed"] + entry["n_open"] - 1
    homo_ev = -hf.matrices["epsilon"][homo_idx] * HARTREE_TO_EV
    assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
        f"Atom {entry['symbol']}: computed Koopmans IE {homo_ev:.3f} eV, "
        f"reference {entry['hf_ie_eV']:.3f} eV "
        f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
    )


# ---------------------------------------------------------------------------
# Tests — ROHF molecules
# ---------------------------------------------------------------------------


def test_rohf_molecule_scf_converged(rohf_molecule_result) -> None:
    """ROHF SCF converges for every open-shell molecule.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_molecule_result
    assert hf.converged, (
        f"ROHF SCF did not converge for {entry['formula']}."
    )


def test_rohf_molecule_total_energy_negative(rohf_molecule_result) -> None:
    """ROHF total energy is negative for every open-shell molecule.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_molecule_result
    assert hf.e_total < 0.0, (
        f"{entry['formula']}: total energy {hf.e_total:.6f} Ha should be negative."
    )


def test_rohf_molecule_homo_ie_positive(rohf_molecule_result) -> None:
    """Koopmans IE from the ROHF HOMO is positive (HOMO is a bound orbital).

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_molecule_result
    homo_idx = entry["n_closed"] + entry["n_open"] - 1
    homo_ev = -hf.matrices["epsilon"][homo_idx] * HARTREE_TO_EV
    assert homo_ev > 0.0, (
        f"{entry['formula']}: Koopmans IE {homo_ev:.3f} eV should be positive."
    )


def test_rohf_molecule_koopmans_ie_vs_reference(rohf_molecule_result) -> None:
    """Koopmans IE from the ROHF HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The HOMO is the last singly-occupied orbital at index
    ``n_closed + n_open − 1`` in the ascending-order effective-Fock
    eigenvalue array.

    :param rohf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = rohf_molecule_result
    homo_idx = entry["n_closed"] + entry["n_open"] - 1
    homo_ev = -hf.matrices["epsilon"][homo_idx] * HARTREE_TO_EV
    assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
        f"{entry['formula']}: computed Koopmans IE {homo_ev:.3f} eV, "
        f"reference {entry['hf_ie_eV']:.3f} eV "
        f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
    )
