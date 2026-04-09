"""Generate expected occupancy maps for all elements.

This script produces two auto-generated Python modules used by the test
suite as golden references:

- ``expected_atom_pure.py`` — full spin-orbital occupancy maps produced by
    the theoretical Aufbau+Hund filling for Z=1..118.
- ``expected_atom_empirical.py`` — deviations from theory where empirical
    exceptions apply; contains only elements whose empirical mapping differs
    from the pure theoretical result.

The generation process builds an ``Atom`` instance for each Z, extracts the
complete spin-orbital occupancy map and serializes the mapping to the
corresponding output file.

Note
----
The empirical file is produced by applying ``EMPIRICAL_EXCEPTIONS`` on top
of the pure theoretical map; generated files are authoritative and should be
reviewed before committing.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

from q_block.models.atom import Atom
from q_block.constants.natural.atoms_data import (
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
    EMPIRICAL_EXCEPTIONS,
)

# Type aliases
SpinKey = Tuple[int, int, int, float]  # (n, l, m, s)
SpinMap = Dict[SpinKey, bool]


def _extract_spin_map_from_atom(atom_instance: Atom) -> SpinMap:
    """Extract a full spin-orbital occupancy map from an Atom instance.

    :param atom_instance: Atom with pre-built shells and occupancy flags set.
    :type atom_instance: Atom

    :returns: Mapping of spin-orbital keys to occupancy booleans.
    :rtype: SpinMap
    """
    mapping: SpinMap = {}
    # Atom provides either atom.shells or atom.spinorbitals; we traverse shells/subshells to be safe
    for shell in atom_instance.shells.values():
        for subshell in shell.subshells:
            for orb in subshell.orbitals:
                mapping[(orb.n, orb.l, orb.m, orb.spin_up.s)] = bool(
                    orb.spin_up.occupied
                )
                mapping[(orb.n, orb.l, orb.m, orb.spin_down.s)] = bool(
                    orb.spin_down.occupied
                )
    return mapping


def _apply_empirical_to_map(
    base_map: SpinMap, instructions: List[Dict]
) -> SpinMap:
    """Apply empirical subshell occupancy instructions onto a base spin map.

    The function returns a new copy of ``base_map`` with occupancy for the
    specified ``(n,l)`` subshells replaced according to the provided
    ``instructions``. Each instruction is a mapping with keys ``"n"``,
    ``"l"`` and ``"electron_count"``.

    :param base_map: The theoretical occupancy map to serve as a base.
    :type base_map: SpinMap
    :param instructions: List of subshell assignment dictionaries.
    :type instructions: List[Dict]

    :returns: New SpinMap with empirical assignments applied.
    :rtype: SpinMap
    """
    out_map = deepcopy(base_map)

    for ins in instructions:
        n = int(ins["n"])
        l = int(ins["l"])
        count = int(ins["electron_count"])

        ms = list(range(-l, l + 1))
        num_orb = len(ms)

        # Determine keys for this subshell in the out_map; if some keys missing,
        # we create them (defensive) so full map covers subshell (n,l).
        for m in ms:
            up_key = (n, l, m, 0.5)
            down_key = (n, l, m, -0.5)
            if up_key not in out_map:
                out_map[up_key] = False
            if down_key not in out_map:
                out_map[down_key] = False

        # Clear existing occupancy for this subshell (we will set according to instruction)
        for m in ms:
            out_map[(n, l, m, 0.5)] = False
            out_map[(n, l, m, -0.5)] = False

        # First pass: spin-up
        first = min(num_orb, count)
        for i in range(first):
            m = ms[i]
            out_map[(n, l, m, 0.5)] = True

        # Second pass: spin-down
        remaining = max(0, count - first)
        second = min(num_orb, remaining)
        for i in range(second):
            m = ms[i]
            out_map[(n, l, m, -0.5)] = True

    return out_map


def _spinmaps_identical(m1: SpinMap, m2: SpinMap) -> bool:
    """Return True when two spin maps have identical keys and values.

    :param m1: First spin map.
    :type m1: SpinMap
    :param m2: Second spin map.
    :type m2: SpinMap

    :returns: ``True`` when maps are identical, otherwise ``False``.
    :rtype: bool
    """
    if set(m1.keys()) != set(m2.keys()):
        return False
    for k in m1:
        if bool(m1[k]) != bool(m2[k]):
            return False
    return True


def _subshell_counts_from_map(spinmap: SpinMap) -> List[Tuple[int, int, int]]:
    """Compute occupied electron counts per subshell from a spin map.

    :param spinmap: Mapping of spin-orbital keys to occupancy booleans.
    :type spinmap: SpinMap

    :returns: Sorted list of ``(n, l, count)`` triples for occupied subshells.
    :rtype: List[Tuple[int, int, int]]
    """
    counts: Dict[Tuple[int, int], int] = {}
    for (n, l, m, s), occ in spinmap.items():
        if occ:
            counts[(n, l)] = counts.get((n, l), 0) + 1
    triples = [(n, l, c) for (n, l), c in counts.items()]
    triples.sort(key=lambda x: (x[0], x[1]))
    return triples


OUT_DIR = Path("./tests/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PURE_FILE = OUT_DIR / "expected_atom_pure.py"
EMP_FILE = OUT_DIR / "expected_atom_empirical.py"

HEADER = """# AUTO-GENERATED FILE
# Generated by generate_expected_atom_data.py
# DO NOT EDIT MANUALLY
#
# Contains expected spin-orbital occupancy maps produced directly from:
#     Atom.fill() (theoretical) and by applying EMPIRICAL_EXCEPTIONS to theory
#
# For each element Z:
#   # Z = {Z}  {Symbol}
#   # config: <configuration string>   (derived from final counts)
#   # subshells: [(n,l,count), ...]
#   EXPECTED_ATOM_PURE[Z] = {{ ... }}
#
"""

PURE_FILE.write_text(HEADER + "EXPECTED_ATOM_PURE = {}\n\n", encoding="utf8")
EMP_FILE.write_text(
    HEADER + "EXPECTED_ATOM_EMPIRICAL = {}\n\n", encoding="utf8"
)


for Z in range(1, 119):
    sym = ATOMS_SYMBOLS_Z_TO_SYMBOL[Z]

    # --- PURE (theory) ---
    atom_pure = Atom(Z=Z, n_max=7, use_empirical_exceptions=False)
    atom_pure.fill()
    spinmap_pure = _extract_spin_map_from_atom(atom_pure)
    subs_pure = _subshell_counts_from_map(spinmap_pure)
    # Build a config string like "1s2 2s2 2p6 ..."
    l_to_letter = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    cfg_parts = [f"{n}{l_to_letter[l]}{count}" for (n, l, count) in subs_pure]
    cfg_pure = " ".join(cfg_parts)

    with PURE_FILE.open("a", encoding="utf8") as f:
        f.write(f"# Z = {Z}   {sym}\n")
        f.write(f"# config: {cfg_pure}\n")
        f.write(f"# subshells: {subs_pure}\n")
        f.write(f"EXPECTED_ATOM_PURE[{Z}] = {{\n")
        for key, val in sorted(spinmap_pure.items()):
            f.write(f"    {key}: {val},\n")
        f.write("}\n\n")

    # --- EMPIRICAL (apply exceptions ON TOP OF THEORY) ---
    # Start from pure map and apply EMPIRICAL_EXCEPTIONS if present
    if Z in EMPIRICAL_EXCEPTIONS:
        ins_list = EMPIRICAL_EXCEPTIONS[Z]
        spinmap_emp = _apply_empirical_to_map(spinmap_pure, ins_list)
        subs_emp = _subshell_counts_from_map(spinmap_emp)
        cfg_parts_emp = [
            f"{n}{l_to_letter[l]}{count}" for (n, l, count) in subs_emp
        ]
        cfg_emp = " ".join(cfg_parts_emp)

        # Only write empirical if it actually differs from pure
        if not _spinmaps_identical(spinmap_pure, spinmap_emp):
            with EMP_FILE.open("a", encoding="utf8") as f:
                f.write(f"# Z = {Z}   {sym}\n")
                f.write(f"# config: {cfg_emp}\n")
                f.write(f"# subshells: {subs_emp}\n")
                f.write(f"EXPECTED_ATOM_EMPIRICAL[{Z}] = {{\n")
                for key, val in sorted(spinmap_emp.items()):
                    f.write(f"    {key}: {val},\n")
                f.write("}\n\n")

print("Done.")
print("Generated: expected_atom_pure.py and expected_atom_empirical.py")
