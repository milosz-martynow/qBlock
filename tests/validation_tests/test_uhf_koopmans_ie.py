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

import pytest

from q_block.methods.wavefunction.hartree_fock.unrestricted_hartree_fock import (
    UnrestrictedHartreeFock,
)
from tests.validation_tests.utils import ABS_TOL_EV, HARTREE_TO_EV, _build_from_geometry
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
    if entry["multiplicity"] > 1
]

# UHF molecules
_uhf_mol_entries = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if entry["method"] == "UHF"
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (each SCF is run once, shared across test functions)
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _uhf_atom_entries],
    ids=[key for key, _ in _uhf_atom_entries],
)
def uhf_atom_result(request):
    """Run UHF SCF for one open-shell atom (placed at origin) and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=[{"symbol": entry["symbol"], "x": 0.0, "y": 0.0, "z": 0.0}],
        multiplicity=entry["multiplicity"],
    )
    hf = UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        max_iterations=200,
    ).run()
    return entry, hf


@pytest.fixture(
    scope="module",
    params=[entry for _, entry in _uhf_mol_entries],
    ids=[key for key, _ in _uhf_mol_entries],
)
def uhf_molecule_result(request):
    """Run UHF SCF for one open-shell molecule and return
    ``(entry, converged_hf)``."""
    entry = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
    )
    hf = UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        n_alpha=entry["n_alpha"],
        n_beta=entry["n_beta"],
        max_iterations=200,
    ).run()
    return entry, hf


# ---------------------------------------------------------------------------
# Tests — open-shell atoms
# ---------------------------------------------------------------------------


def test_uhf_atom_scf_converged(uhf_atom_result) -> None:
    """UHF SCF converges for every open-shell atom.

    :param uhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_atom_result
    assert hf.converged, (
        f"UHF SCF did not converge for atom {entry['symbol']} "
        f"(Z={entry['n_electrons']})."
    )


def test_uhf_atom_koopmans_ie_vs_reference(uhf_atom_result) -> None:
    """Koopmans IE from the UHF alpha HOMO agrees with ``hf_ie_eV``.

    The alpha HOMO is at index ``n_alpha − 1`` in the ascending-order
    alpha-spin eigenvalue array ``epsilon_alpha``.

    :param uhf_atom_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_atom_result
    homo_idx = entry["n_alpha"] - 1
    homo_ev = -hf.matrices["epsilon_alpha"][homo_idx] * HARTREE_TO_EV
    assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
        f"Atom {entry['symbol']}: UHF Koopmans IE {homo_ev:.3f} eV, "
        f"reference {entry['hf_ie_eV']:.3f} eV "
        f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
    )


# ---------------------------------------------------------------------------
# Tests — UHF molecules
# ---------------------------------------------------------------------------


def test_uhf_molecule_scf_converged(uhf_molecule_result) -> None:
    """UHF SCF converges for every open-shell molecule.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_molecule_result
    assert hf.converged, (
        f"UHF SCF did not converge for {entry['formula']}."
    )


def test_uhf_molecule_total_energy_negative(uhf_molecule_result) -> None:
    """UHF total energy is negative for every open-shell molecule.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_molecule_result
    assert hf.e_total < 0.0, (
        f"{entry['formula']}: total energy {hf.e_total:.6f} Ha should be negative."
    )


def test_uhf_molecule_alpha_homo_ie_positive(uhf_molecule_result) -> None:
    """Koopmans IE from the UHF alpha HOMO is positive (HOMO is a bound orbital).

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_molecule_result
    homo_idx = entry["n_alpha"] - 1
    homo_ev = -hf.matrices["epsilon_alpha"][homo_idx] * HARTREE_TO_EV
    assert homo_ev > 0.0, (
        f"{entry['formula']}: UHF Koopmans IE {homo_ev:.3f} eV should be positive."
    )


def test_uhf_molecule_koopmans_ie_vs_reference(uhf_molecule_result) -> None:
    """Koopmans IE from the UHF alpha HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV.

    The alpha HOMO is at index ``n_alpha − 1`` in the ascending-order
    ``epsilon_alpha`` array.

    :param uhf_molecule_result: Pytest fixture providing ``(entry, hf)``.
    """
    entry, hf = uhf_molecule_result
    homo_idx = entry["n_alpha"] - 1
    homo_ev = -hf.matrices["epsilon_alpha"][homo_idx] * HARTREE_TO_EV
    assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
        f"{entry['formula']}: UHF Koopmans IE {homo_ev:.3f} eV, "
        f"reference {entry['hf_ie_eV']:.3f} eV "
        f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
    )
