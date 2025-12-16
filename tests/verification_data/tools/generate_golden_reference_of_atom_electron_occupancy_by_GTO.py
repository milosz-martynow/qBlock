"""
Golden-reference generator for GTO population tests.

===============================================================================
PURPOSE
===============================================================================

This script generates *golden-reference snapshots* for the function:

    Atom.populate_spinorbitals_with_gto

Golden references are authoritative, version-controlled data files that
represent the *correct and verified* mapping between:

    (Atom, Basis Set) → occupied subshells → GTO parameters

They are used by unit tests to detect *any regression* in the mapping logic.

-------------------------------------------------------------------------------
IMPORTANT WARNING
-------------------------------------------------------------------------------

Golden references MUST NOT be regenerated casually.

After running this script:

  1. The generated JSON files MUST be reviewed manually
  2. Values MUST be validated against:
       - the corresponding .gbs basis file
       - expected quantum numbers (n, l)
       - known basis-set structure (split-valence, diffuse, polarization)
  3. Only after verification should the files be committed

Blind regeneration defeats the purpose of golden-reference testing.

===============================================================================
GOLDEN REFERENCE ARCHITECTURE
===============================================================================

Golden references are stored under:

    tests/verification_data/gto_population/

Each basis set produces exactly ONE JSON file:

    gto_population/
      3-21G.json
      6-31G.json
      6-311G.json
      6-311++Gss.json

Each file has the structure:

    {
      "basis": "6-31G",
      "atoms": {
        "H": {
          "symbol": "H",
          "Z": 1,
          "basis": "6-31G",
          "orbitals": { ... }
        },
        "C": { ... },
        "Fe": { ... }
      }
    }

-------------------------------------------------------------------------------
WHAT IS *NOT* STORED
-------------------------------------------------------------------------------

- region names (core / valence)
- AO indices
- normalization constants
- implementation-specific ordering

===============================================================================
HOW TO USE
===============================================================================

Run from project root:

    python tools/generate_golden.py

===============================================================================
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from q_block.atom import Atom
from q_block.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z
from q_block.basis_set_pople import parse_gaussian_basis
from tests.test_atom import _serialize_atom_for_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASIS_ROOT: Path = Path("./data/basis_set/gto_gaussian_format")

GOLDEN_ROOT: Path = Path("./tests/verification_data/gto_population")

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
    """
    Generate golden-reference JSON snapshots for all atoms
    in all configured Gaussian basis sets.

    Output:
        One JSON file per basis set, containing all atoms.
    """

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)

    for filename in BASIS_FILES:

        basis_path: Path = BASIS_ROOT / filename
        basis = parse_gaussian_basis(filepath=str(basis_path))

        basis_name: str = basis_path.stem

        atoms_data: Dict[str, Dict[str, Any]] = OrderedDict()

        for symbol, atom_basis in basis.items():
            atom = Atom(
                Z=ATOMS_SYMBOLS_SYMBOL_TO_Z[symbol],
                basis_set=atom_basis,
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
