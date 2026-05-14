"""
Pople Basis Sets
================

Lazily-loaded :class:`~q_block.compute.environment.io.basis_set.Pople`
instances for every Gaussian-type-orbital Pople basis set bundled with
qBlock.  Each basis set object is parsed from disk only on first access
and then cached for subsequent uses.

Usage::

    from q_block.compute.environment.constants.numerical.pople import G631pp    # 6-31ppG  (was 6-31++G)
    from q_block.compute.environment.constants.numerical.pople import G6311ppss # 6-311ppGss (was 6-311++G**)
    from q_block.compute.environment.constants.numerical.pople import STO3G     # STO-3G

Naming Convention
-----------------
* ``G`` prefix for all split-valence Gaussian-style sets.
* ``STO`` prefix for Slater-type orbital minimal basis sets.
* Numbers encode the split: ``631`` for 6-31, ``6311`` for 6-311, etc.
* ``p`` replaces one ``+`` diffuse-function marker; ``pp`` replaces ``++``.
* ``s`` denotes a single polarization shell (``*``); ``ss`` for ``**``.
* ``J`` and ``RIFIT`` suffixes are preserved for auxiliary / RI-fitting sets.
* Parenthesised polarization specs are flattened: ``(d,p)`` → ``dp``,
  ``(2df,p)`` → ``2dfp``, ``(2df,2pd)`` → ``2df2pd``, etc.

Available Names
---------------
=================  ========================
Python name        Basis set file
=================  ========================
``STO3G``          STO-3G.gbs
``STO6G``          STO-6G.gbs
``G321``           3-21G.gbs
``G431``           4-31G.gbs
``G521``           5-21G.gbs
``G621``           6-21G.gbs
``G631``           6-31G.gbs
``G631s``          6-31Gs.gbs
``G631ss``         6-31Gss.gbs
``G631ssRIFIT``    6-31Gss-RIFIT.gbs
``G631J``          6-31G-J.gbs
``G631dp``         6-31G(d,p).gbs
``G6312dfp``       6-31G(2df,p).gbs
``G6313df3pd``     6-31G(3df,3pd).gbs
``G631p``          6-31pG.gbs
``G631ps``         6-31pGs.gbs
``G631pss``        6-31pGss.gbs
``G631psJ``        6-31pGs-J.gbs
``G631pp``         6-31ppG.gbs
``G631pps``        6-31ppGs.gbs
``G631ppss``       6-31ppGss.gbs
``G631ppssJ``      6-31ppGss-J.gbs
``G6311``          6-311G.gbs
``G6311s``         6-311Gs.gbs
``G6311ss``        6-311Gss.gbs
``G6311ssRIFIT``   6-311Gss-RIFIT.gbs
``G6311J``         6-311G-J.gbs
``G6311dp``        6-311G(d,p).gbs
``G63112df2pd``    6-311G(2df,2pd).gbs
``G6311p``         6-311pG.gbs
``G6311ps``        6-311pGs.gbs
``G6311pss``       6-311pGss.gbs
``G6311psJ``       6-311pGs-J.gbs
``G6311p2dp``      6-311pG(2d,p).gbs
``G6311pp``        6-311ppG.gbs
``G6311pps``       6-311ppGs.gbs
``G6311ppss``      6-311ppGss.gbs
``G6311ppssJ``     6-311ppGss-J.gbs
``G6311pp2d2p``    6-311ppG(2d,2p).gbs
``G6311pp3df3pd``  6-311ppG(3df,3pd).gbs
=================  ========================
"""

import importlib.resources
from pathlib import Path
from typing import Dict

from q_block.compute.environment.io.basis_set import Pople

# ---------------------------------------------------------------------------
# Path to the bundled Pople basis-set directory
# ---------------------------------------------------------------------------

BASIS_DIR: Path = (
    Path(
        str(
            importlib.resources.files(
                "q_block.compute.environment.constants.numerical"
            )
        )
    )
    / "basis_set"
    / "pople"
)

# ---------------------------------------------------------------------------
# Mapping: Python identifier  →  filename inside BASIS_DIR
# ---------------------------------------------------------------------------

_NAMES: Dict[str, str] = {
    # ── Slater-type orbital minimal sets ───────────────────────────────────
    "STO3G":         "STO-3G.gbs",
    "STO6G":         "STO-6G.gbs",
    # ── Split-valence (no diffuse, no polarization) ────────────────────────
    "G321":          "3-21G.gbs",
    "G431":          "4-31G.gbs",
    "G521":          "5-21G.gbs",
    "G621":          "6-21G.gbs",
    "G631":          "6-31G.gbs",
    "G6311":         "6-311G.gbs",
    # ── Polarized (no diffuse) ─────────────────────────────────────────────
    "G631s":         "6-31Gs.gbs",
    "G631ss":        "6-31Gss.gbs",
    "G631dp":        "6-31G(d,p).gbs",
    "G6312dfp":      "6-31G(2df,p).gbs",
    "G6313df3pd":    "6-31G(3df,3pd).gbs",
    "G6311s":        "6-311Gs.gbs",
    "G6311ss":       "6-311Gss.gbs",
    "G6311dp":       "6-311G(d,p).gbs",
    "G63112df2pd":   "6-311G(2df,2pd).gbs",
    # ── Single diffuse (+) ─────────────────────────────────────────────────
    "G631p":         "6-31pG.gbs",
    "G631ps":        "6-31pGs.gbs",
    "G631pss":       "6-31pGss.gbs",
    "G6311p":        "6-311pG.gbs",
    "G6311ps":       "6-311pGs.gbs",
    "G6311pss":      "6-311pGss.gbs",
    "G6311p2dp":     "6-311pG(2d,p).gbs",
    # ── Double diffuse (++) ────────────────────────────────────────────────
    "G631pp":        "6-31ppG.gbs",
    "G631pps":       "6-31ppGs.gbs",
    "G631ppss":      "6-31ppGss.gbs",
    "G6311pp":       "6-311ppG.gbs",
    "G6311pps":      "6-311ppGs.gbs",
    "G6311ppss":     "6-311ppGss.gbs",
    "G6311pp2d2p":   "6-311ppG(2d,2p).gbs",
    "G6311pp3df3pd": "6-311ppG(3df,3pd).gbs",
    # ── Auxiliary / RI-fitting sets ────────────────────────────────────────
    "G631J":         "6-31G-J.gbs",
    "G631psJ":       "6-31pGs-J.gbs",
    "G631ppssJ":     "6-31ppGss-J.gbs",
    "G631ssRIFIT":   "6-31Gss-RIFIT.gbs",
    "G6311J":        "6-311G-J.gbs",
    "G6311psJ":      "6-311pGs-J.gbs",
    "G6311ppssJ":    "6-311ppGss-J.gbs",
    "G6311ssRIFIT":  "6-311Gss-RIFIT.gbs",
}

__all__ = ["BASIS_DIR"] + list(_NAMES.keys())

# ---------------------------------------------------------------------------
# Lazy-loading cache
# ---------------------------------------------------------------------------

_cache: Dict[str, Pople] = {}


def __getattr__(name: str) -> Pople:
    """Return the :class:`~q_block.compute.environment.io.basis_set.Pople`
    instance for *name*, parsing the file on first access.

    :param name: Python identifier for the requested basis set (e.g.
        ``"G631pp"``).
    :type name: str

    :returns: Parsed Pople basis set object.
    :rtype: Pople

    :raises AttributeError: If *name* is not a known basis set identifier.
    """
    if name not in _NAMES:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            f"Available basis sets: {', '.join(sorted(_NAMES))}."
        )
    if name not in _cache:
        _cache[name] = Pople(filepath=str(BASIS_DIR / _NAMES[name]))
    return _cache[name]
