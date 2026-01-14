"""
Expanded Gaussian Basis Set Parser
=================================

The expanded basis set is represented as a hierarchical dictionary with
the following structure:

BasisSet
└── element (str)
    └── regions
        ├── "core"
        ├── "valence_inner"
        └── "valence_outer"
            └── l (int, angular momentum quantum number)
                └── List[BasisSetParameters]
                    ├── "exponents": List[float]
                    └── "coefficients": List[float]

Notes
-----
* Angular momentum is encoded as a dictionary key (l = 0, 1, 2, ...)
* Each shell contains one contraction coefficient per primitive
  (SP shells are expanded beforehand)
* Core / valence-inner / valence-outer regions follow Pople basis semantics
"""

import re
from typing import Any, Dict, List, Literal, Tuple, TypeAlias

from q_block.constants.atoms_data import ANGULAR_MOMENTUM_MAP

RegionName: TypeAlias = Literal[
    "core",
    "valence_inner",
    "valence_outer",
]

BasisSetParameters: TypeAlias = Dict[
    Literal["exponents", "coefficients"],
    List[float],
]

AngularMomentumShells: TypeAlias = Dict[
    int,
    List[BasisSetParameters],
]

Regions: TypeAlias = Dict[
    RegionName,
    AngularMomentumShells,
]

BasisSet: TypeAlias = Dict[
    str,  # element symbol
    Regions,
]


def _parse_float(token: str) -> float:
    """
    Parse a floating-point value from Gaussian/Fortran numeric notation.

    This helper function converts Fortran-style scientific notation
    using ``D`` or ``d`` (e.g. ``1.234D+02``) into standard Python
    scientific notation using ``E`` before converting to ``float``.

    :param token: String representation of a floating-point number. The
        value may use Fortran-style exponent notation (``D`` or ``d``)
        or standard scientific notation (``E``).
    :type token: str

    :returns: The parsed floating-point value.
    :rtype: float
    """
    return float(token.replace("D", "E").replace("d", "E"))


def _format_shell(shell: Dict[str, Any]) -> BasisSetParameters:
    """
    Convert an internal shell representation into a clean, serializable
    dictionary.

    :param shell: Shell dictionary containing primitives.
    :type shell: Dict[str, Any]

    :returns: Dictionary with number of primitives, exponents, and
        coefficients.
    :rtype: Dict[str, Any]
    """
    return {
        "exponents": [p[0] for p in shell["primitives"]],
        "coefficients": [p[1][0] for p in shell["primitives"]],
    }


def _build_regions(
    raw_shells: List[Dict[str, Any]],
) -> Regions:
    """
    Convert parsed Gaussian shells into a structured dictionary with
    numerical angular momentum and region classification.

    :param raw_shells: Shells as parsed directly from the Gaussian basis
        file.
    :type raw_shells: List[Dict[str, Any]]

    :returns: Basis set dictionary split into core, valence-inner, and
        valence-outer regions and grouped by angular momentum.
    :rtype: Regions
    """
    shells: List[Dict[str, Any]] = []

    # Expand combined SP shells into separate S and P shells
    for sh in raw_shells:
        if sh["type"] == "SP":
            s_prims = [(e, [c[0]]) for e, c in sh["primitives"]]
            p_prims = [(e, [c[1]]) for e, c in sh["primitives"]]
            shells.append({"l": 0, "primitives": s_prims})
            shells.append({"l": 1, "primitives": p_prims})
        else:
            shells.append(
                {
                    "l": ANGULAR_MOMENTUM_MAP[sh["type"]],
                    "primitives": sh["primitives"],
                }
            )

    # Group shells by angular momentum
    by_l: Dict[int, List[Dict[str, Any]]] = {}
    for sh in shells:
        l = sh["l"]
        if l not in by_l:
            by_l[l] = []
        by_l[l].append(sh)

    regions: Regions = {
        "core": {},
        "valence_inner": {},
        "valence_outer": {},
    }

    # Assign shells to regions based on tightness ordering
    for l, sh_list in by_l.items():
        sh_list.sort(
            key=lambda sh: max(p[0] for p in sh["primitives"]),
            reverse=True,
        )

        if l == 0:
            if sh_list:
                regions["core"].setdefault(l, []).append(
                    _format_shell(sh_list[0])
                )
            if len(sh_list) >= 2:
                regions["valence_inner"].setdefault(l, []).append(
                    _format_shell(sh_list[1])
                )
            for sh in sh_list[2:]:
                regions["valence_outer"].setdefault(l, []).append(
                    _format_shell(sh)
                )
        else:
            if sh_list:
                regions["valence_inner"].setdefault(l, []).append(
                    _format_shell(sh_list[0])
                )
            for sh in sh_list[1:]:
                regions["valence_outer"].setdefault(l, []).append(
                    _format_shell(sh)
                )

    return regions


def parse_gaussian_basis(filepath: str) -> BasisSet:
    """
    Parse a Gaussian-format basis set file (.gbs) and convert it into a
    structured dictionary with angular momentum separation and region
    classification.

    :param filepath: Path to the Gaussian-format basis set file.
    :type filepath: str

    :returns: Dictionary keyed by atomic symbol, containing fully
        expanded basis set data grouped by region and angular momentum.
        See Header of this script for more data structure architecture.
    :rtype: BasisSet
    """
    with open(filepath, "r") as f:
        lines: List[str] = [ln.strip() for ln in f if ln.strip()]

    basis_by_element: BasisSet = {}
    i: int = 0

    while i < len(lines):
        line = lines[i]

        # Element header (e.g. "C 0")
        if re.match(r"^[A-Z][a-z]?\s+0$", line):
            element = line.split()[0]
            raw_shells: List[Dict[str, Any]] = []
            i += 1

            while i < len(lines):
                line = lines[i]

                match = re.match(r"^(SP|[SPDFG])\s+(\d+)", line)
                if match:
                    shell_type = match.group(1)
                    nprim = int(match.group(2))
                    i += 1

                    primitives: List[Tuple[float, List[float]]] = []
                    for _ in range(nprim):
                        parts = lines[i].split()
                        exponent = _parse_float(parts[0])
                        coefficients = [_parse_float(c) for c in parts[1:]]
                        primitives.append((exponent, coefficients))
                        i += 1

                    raw_shells.append(
                        {
                            "type": shell_type,
                            "primitives": primitives,
                        }
                    )
                    continue

                if line == "****":
                    i += 1
                    break

                i += 1

            basis_by_element[element] = _build_regions(raw_shells)
            continue

        i += 1

    if not basis_by_element:
        raise ValueError("No elements found in basis file.")

    return basis_by_element

