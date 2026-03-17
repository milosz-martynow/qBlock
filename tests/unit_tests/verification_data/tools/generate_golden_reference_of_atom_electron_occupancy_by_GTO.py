"""Generate golden-reference JSON snapshots for GTO population tests.

This module creates one JSON snapshot per Gaussian basis set containing the
verified mapping between occupied subshells and Gaussian-type orbital (GTO)
parameters for all atoms present in the basis. The generated files live
under ``tests/verification_data/gto_population/``.

Warning
-------
Regenerate golden references only after careful manual review and validation
against the source ``.gbs`` basis files.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from q_block.models.atom import Atom
from q_block.constants.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z
from q_block.io.basis_set import Pople
from tests.unit_tests.models.test_atom import _serialize_atom_for_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")

GOLDEN_ROOT: Path = Path("./tests/unit_tests/verification_data/gto_population")

BASIS_FILES: List[str] = [
    "3-21G.gbs",
    "6-31G.gbs",
    "6-311G.gbs",
    "6-311++Gss.gbs",
]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_golden() -> None:
    """Generate golden-reference JSON snapshots for configured basis sets.

    The function parses each configured Gaussian basis file, constructs an
    ``Atom`` for every element present in the basis, calls
    :meth:`Atom.populate_spinorbitals_with_gto` and serializes the resulting
    per-atom data into a single JSON file per basis.

    :returns: None
    :rtype: None
    """

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)

    for filename in BASIS_FILES:

        basis_path: Path = BASIS_ROOT / filename
        basis = Pople(filepath=str(basis_path))

        basis_name: str = basis_path.stem

        atoms_data: Dict[str, Dict[str, Any]] = OrderedDict()

        for symbol in basis.elements:
            atom = Atom(
                Z=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
                basis_set=basis,
            )
            atom.populate_spinorbitals_with_gto()

            atoms_data[symbol] = _serialize_atom_for_test(
                atom=atom,
                basis_name=basis_name,
            )

        out_file: Path = GOLDEN_ROOT / f"{basis_name}.json"
        with out_file.open("w") as f:
            json.dump(
                {
                    "basis": basis_name,
                    "atoms": atoms_data,
                },
                f,
                indent=2,
            )

        print(f"[OK] Generated {out_file}")


if __name__ == "__main__":
    generate_golden()
