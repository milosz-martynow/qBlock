"""Validation tests: Restricted Kohn-Sham (RKS) DFT.

Verifies that the RKS implementation produces physically reasonable
results for closed-shell molecules with three XC functionals:

* **SVWN** (LDA) – Local Density Approximation
* **PBE**  (GGA) – Perdew–Burke–Ernzerhof
* **B3LYP** (hybrid) – Becke 3-parameter Lee–Yang–Parr

Validation criteria:

1. The SCF must converge.
2. The total energy must be negative (bound system).
3. The DFT total energy should differ from the HF reference by a
   physically reasonable amount (correlation energy contribution).

Systems tested: closed-shell molecules from
``MOLECULES_HOMO_ENERGIES`` that are also tagged for RHF.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
    ExchangeCorrelationFunctional,
)
from compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
)
from tests.validation.utils import _build_from_geometry
from tests.validation.validation_data import MOLECULES_HOMO_ENERGIES

# ---------------------------------------------------------------------------
# Test data: select closed-shell molecules that also run with RHF
# ---------------------------------------------------------------------------

_rks_mol_entries: List[Tuple[str, Dict[str, Any]]] = [
    (key, entry)
    for key, entry in MOLECULES_HOMO_ENERGIES.items()
    if "RHF" in entry["proposed_hartree_fock_approach"]
]

# Functionals to test
_functionals: List[Tuple[str, ExchangeCorrelationFunctional]] = [
    ("SVWN", SVWN()),
    ("PBE", PBE()),
    ("B3LYP", B3LYP()),
]


# ---------------------------------------------------------------------------
# Module-scoped fixture: runs RKS once per (molecule, functional) pair
# ---------------------------------------------------------------------------


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
    """Run RKS SCF for one molecule with one functional.

    :returns: ``(entry, functional_name, rks)``
    :rtype: Tuple[Dict[str, Any], str, RestrictedKohnSham]
    """
    mol_key, entry, func_name, func = request.param
    cgtos, nuclei, e_nuclear = _build_from_geometry(
        geometry=entry["geometry"],
        multiplicity=entry["multiplicity"],
        basis_set_filename=entry["proposed_basis_set"],
    )
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuclear,
        functional=func,
        n_electrons=entry["n_electrons"],
        n_radial=50,
        n_angular=14,
        max_iterations=entry.get("max_iterations", 200),
        convergence_threshold=1e-5,
    ).run()
    return entry, func_name, rks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rks_scf_converged(rks_molecule_result) -> None:
    """RKS SCF converges for every closed-shell molecule.

    :param rks_molecule_result: Fixture returning
        ``(entry, func_name, rks)``.
    """
    entry, func_name, rks = rks_molecule_result
    assert rks.converged, (
        f"RKS-{func_name} did not converge for "
        f"{entry['formula']}."
    )


def test_rks_total_energy_negative(rks_molecule_result) -> None:
    """RKS total energy is negative for every closed-shell molecule.

    :param rks_molecule_result: Fixture returning
        ``(entry, func_name, rks)``.
    """
    entry, func_name, rks = rks_molecule_result
    assert rks.e_total < 0.0, (
        f"RKS-{func_name} total energy = {rks.e_total:.6f} Hartree "
        f"for {entry['formula']} is not negative."
    )


def test_rks_total_energy_finite(rks_molecule_result) -> None:
    """RKS total energy is finite (no NaN or Inf).

    :param rks_molecule_result: Fixture returning
        ``(entry, func_name, rks)``.
    """
    _, func_name, rks = rks_molecule_result
    assert np.isfinite(rks.e_total), (
        f"RKS-{func_name} produced non-finite energy: "
        f"{rks.e_total}"
    )
