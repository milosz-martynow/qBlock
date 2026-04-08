"""
Basis Set Classes
=================

This module provides a generic :class:`BasisSet` base class and concrete
implementations for different basis set families.

Currently supported basis set types:

* :class:`Pople` – Pople-style Gaussian basis sets, including:
  
  - STO-nG minimal basis sets (STO-3G, STO-6G) – n Gaussians approximating
    Slater-type orbitals
  - Split-valence basis sets (3-21G, 6-31G, 6-311G, 6-311++G**) – different
    numbers of Gaussians for core and valence regions

Future basis set families (Dunning, Ahlrichs, etc.) should inherit from
:class:`BasisSet` and implement :meth:`BasisSet.parse`.

Expanded Gaussian Basis Set Representation
------------------------------------------

The expanded basis set is represented as a hierarchical dictionary with
the following structure:

BasisSetData
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
from typing import Any, Dict, List, Literal, Tuple, TypeAlias, Union

from q_block.constants.atoms_data import (
    ANGULAR_MOMENTUM_MAP,
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)


# ======================================================================
# Type aliases
# ======================================================================

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

BasisSetData: TypeAlias = Dict[
    str,  # element symbol
    Regions,
]


# ======================================================================
# G E N E R I C   B A S I S   S E T   (abstract base)
# ======================================================================

class BasisSet:
    """Base class for all basis set families.

    Every concrete basis set class must override :meth:`parse`, which
    reads a basis set file and returns the parsed data as a
    :data:`BasisSetData` dictionary.

    :param filepath: Path to the basis set file.
    :type filepath: str

    Attributes
    ----------
    filepath : str
        Path to the basis set file.
    data : BasisSetData
        Parsed basis set data keyed by element symbol.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.data: BasisSetData = self.parse(filepath)

    @staticmethod
    def parse(filepath: str) -> "BasisSetData":
        """Parse a basis set file and return structured data.

        Subclasses **must** override this method.

        :param filepath: Path to the basis set file.
        :type filepath: str

        :returns: Dictionary keyed by atomic symbol, containing the
            fully expanded basis set data.
        :rtype: BasisSetData

        :raises NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            "Subclasses must implement the parse() method."
        )

    def __getitem__(self, key: Union[str, int]) -> Regions:
        """Retrieve basis set data for a specific element.

        :param key: Atomic symbol (e.g. ``"H"``, ``"O"``) **or**
            atomic number (e.g. ``1``, ``8``).  When an ``int`` is
            given, the symbol is resolved via
            :data:`~q_block.constants.atoms_data.ATOMS_SYMBOLS_Z_TO_SYMBOL`.
        :type key: Union[str, int]

        :returns: Regions dictionary for the requested element.
        :rtype: Regions

        :raises KeyError: If element is not present in the basis set.
        """
        if isinstance(key, int):
            key = ATOMS_SYMBOLS_Z_TO_SYMBOL[key]
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        """Check if an element is present in this basis set.

        *key* may be an atomic symbol (``str``) or atomic number
        (``int``).
        """
        if isinstance(key, int):
            key = ATOMS_SYMBOLS_Z_TO_SYMBOL.get(key, "")
        return key in self.data

    def __repr__(self) -> str:
        elements = ", ".join(sorted(self.data.keys()))
        return (
            f"{self.__class__.__name__}(filepath={self.filepath!r}, "
            f"elements=[{elements}])"
        )

    @property
    def elements(self) -> List[str]:
        """List of element symbols present in this basis set."""
        return list(self.data.keys())


# ======================================================================
# P O P L E   B A S I S   S E T
# ======================================================================

class Pople(BasisSet):
    """Pople-style Gaussian basis sets.

    This class parses Gaussian-format ``.gbs`` files for Pople basis
    sets and stores the result as an element-keyed dictionary with region
    classification (core / valence-inner / valence-outer).
    
    Supported basis sets include:
    
    * **STO-nG** (minimal basis): n Gaussian primitives approximate each
      Slater-type orbital (e.g., STO-3G, STO-6G)
    * **Split-valence** (3-21G, 6-31G, 6-311G, etc.): Different numbers of
      Gaussian primitives for core vs. valence orbitals
    * **Polarization and diffuse** (6-311++G**, etc.): Extended split-valence
      with additional polarization and diffuse functions

    :param filepath: Path to the Gaussian-format ``.gbs`` file.
    :type filepath: str

    Example
    -------
    >>> basis = Pople("data/basis_set/pople/3-21G.gbs")
    >>> hydrogen_regions = basis["H"]
    """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _format_shell(shell: Dict[str, Any]) -> BasisSetParameters:
        """
        Convert an internal shell representation into a clean, serializable
        dictionary.

        :param shell: Shell dictionary containing primitives.
        :type shell: Dict[str, Any]

        :returns: Dictionary with exponents and coefficients.
        :rtype: BasisSetParameters
        """
        return {
            "exponents": [p[0] for p in shell["primitives"]],
            "coefficients": [p[1][0] for p in shell["primitives"]],
        }

    @staticmethod
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
                        Pople._format_shell(sh_list[0])
                    )
                if len(sh_list) >= 2:
                    regions["valence_inner"].setdefault(l, []).append(
                        Pople._format_shell(sh_list[1])
                    )
                for sh in sh_list[2:]:
                    regions["valence_outer"].setdefault(l, []).append(
                        Pople._format_shell(sh)
                    )
            else:
                if sh_list:
                    regions["valence_inner"].setdefault(l, []).append(
                        Pople._format_shell(sh_list[0])
                    )
                for sh in sh_list[1:]:
                    regions["valence_outer"].setdefault(l, []).append(
                        Pople._format_shell(sh)
                    )

        return regions

    # ------------------------------------------------------------------
    # Public interface (abstract method implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def parse(filepath: str) -> BasisSetData:
        """
        Parse a Gaussian-format basis set file (.gbs) and convert it into a
        structured dictionary with angular momentum separation and region
        classification.

        :param filepath: Path to the Gaussian-format basis set file.
        :type filepath: str

        :returns: Dictionary keyed by atomic symbol, containing fully
            expanded basis set data grouped by region and angular momentum.
            See module docstring for the data structure architecture.
        :rtype: BasisSetData
        """
        with open(filepath, "r") as f:
            lines: List[str] = [ln.strip() for ln in f if ln.strip()]

        basis_by_element: BasisSetData = {}
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
                            exponent = Pople._parse_float(parts[0])
                            coefficients = [
                                Pople._parse_float(c) for c in parts[1:]
                            ]
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

                basis_by_element[element] = Pople._build_regions(raw_shells)
                continue

            i += 1

        if not basis_by_element:
            raise ValueError("No elements found in basis file.")

        return basis_by_element

