"""Golden-reference tests for GTO population via Molecule.

This module validates that :class:`q_block.molecule.Molecule` correctly
populates Gaussian-type orbital (GTO) data on :class:`q_block.atom.Atom`
instances using the same golden-reference data as the original
Atom-level implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pandas as pd
import pytest

from q_block.atom import Atom
from q_block.atoms_data import (
    ATOMS_SYMBOLS_SYMBOL_TO_Z,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)
from q_block.basis_set_pople import parse_gaussian_basis
from q_block.input_data import InputData
from q_block.molecule import Molecule

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")
GOLDEN_ROOT: Path = Path("./tests/verification_data/gto_population")

BASIS_FILES = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311++Gss.gbs",
]


def _serialize_atom_for_test(atom: Atom, basis_name: str) -> Dict[str, Any]:
    """Serialize Atom into golden-reference comparable structure.

    This helper mirrors the structure used in the original Atom tests
    and is intentionally duplicated here to keep the Molecule tests
    self-contained.
    """

    from q_block.electron import Shell, SpinOrbital  # local import to avoid cycles

    orbitals: Dict[str, Dict[str, list]] = {}

    shells: Dict[int, Shell] = atom.shells
    for shell in shells.values():
        for subshell in shell.subshells:

            occupied_spinorbitals = [
                so
                for orb in subshell.orbitals
                for so in (orb.spin_up, orb.spin_down)
                if so.occupied
            ]

            if not occupied_spinorbitals:
                continue

            key = f"{subshell.n}{'spdf'[subshell.l]}"
            so: SpinOrbital = occupied_spinorbitals[0]

            orbitals[key] = {
                "exponents": so.data["exponents"],
                "contractions": so.data["contractions"],
            }

    return {
        "symbol": ATOMS_SYMBOLS_Z_TO_SYMBOL[atom.atomic_number],
        "Z": atom.atomic_number,
        "basis": basis_name,
        "orbitals": orbitals,
    }


def _iter_golden_test_cases() -> Iterator[Tuple[str, str]]:
    """Generate (basis_file, atom_symbol) pairs for pytest parametrization."""

    for basis_file in BASIS_FILES:
        basis_name = Path(basis_file).stem
        golden_path = GOLDEN_ROOT / f"{basis_name}.json"

        with golden_path.open() as f:
            golden = json.load(f)

        for symbol in golden["atoms"]:
            yield basis_file, symbol


@pytest.mark.parametrize(
    "basis_file, symbol",
    list(_iter_golden_test_cases()),
    ids=lambda p: p if isinstance(p, str) else None,
)
def test_molecule_golden_gto_population(basis_file: str, symbol: str) -> None:
    """Verify GTO population via Molecule for a single atom.

    For each (basis_file, symbol) pair, this test constructs a single
    :class:`Atom` with the corresponding basis-set fragment, wraps it in
    :class:`InputData`, and builds a :class:`Molecule`. The Molecule
    initialisation triggers GTO population on the atom, and the
    resulting structure is compared against the golden-reference data.

    :param basis_file: Gaussian basis set filename.
    :type basis_file: str

    :param symbol: Atomic symbol (e.g. ``"H"``, ``"C"``, ``"Fe"``).
    :type symbol: str
    """

    basis_path = BASIS_ROOT / basis_file
    basis_name = basis_path.stem

    golden_path = GOLDEN_ROOT / f"{basis_name}.json"
    with golden_path.open() as f:
        golden = json.load(f)

    basis = parse_gaussian_basis(filepath=str(basis_path))
    golden_atom = golden["atoms"][symbol]

    atom = Atom(
        atomic_number=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
        basis_set=basis[symbol],
    )

    row = {
        "atom_id": "1",
        "atomic_number": atom.atomic_number,
        "symbol": symbol,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "atom": atom,
    }
    input_data = InputData(atoms=pd.DataFrame([row]))

    molecule = Molecule(input_data=input_data)
    assert len(molecule.atoms) == 1

    snapshot = _serialize_atom_for_test(atom=atom, basis_name=basis_name)

    assert snapshot == golden_atom, (
        f"GTO population mismatch for atom {symbol} " f"in basis {basis_name}"
    )
