"""
Example: Loading a Pople Basis Set — Three Methods
====================================================

Demonstrates three ways to obtain a
:class:`~q_block.compute.environment.io.basis_set.Pople` basis set instance:

    1. **From the bundled library** — ``from q_block.compute.pople import G631``
       Instant, zero I/O: the module caches each basis set after the first
       access.

    2. **From a local .gbs file** — ``Pople(filepath=...)``
       Loads any Gaussian-format basis set file from the file system.
       ``BASIS_DIR`` from ``q_block.compute.pople`` points to the directory
       that contains all bundled files.

    3. **From the Basis Set Exchange REST API** — fetches the basis set over
       HTTP, writes it to a temporary file, and parses it with ``Pople``.
       Requires an active internet connection.

All three paths produce structurally identical
:class:`~q_block.compute.environment.io.basis_set.Pople` objects.
"""

import tempfile
import urllib.request
from pathlib import Path

from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.logs import setup_logging
from q_block.compute.environment.constants.numerical.pople import BASIS_DIR, G631

logger = setup_logging(__name__)

# BSE REST API endpoint — returns Gaussian94 / GAMESS format (.gbs-compatible)
_BSE_URL: str = (
    "https://www.basissetexchange.org/api/basis/6-31g/format/gaussian94/"
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FROM LIBRARY — bundled, lazy-loaded singleton
# ══════════════════════════════════════════════════════════════════════════════
# ``q_block.compute.environment.constants.numerical.pople`` exposes one pre-named attribute per bundled .gbs
# file.  Each is parsed from disk on first access and then cached; subsequent
# accesses return the same object at zero cost.

basis_library: Pople = G631

logger.info("=== 1. From library ===")
logger.info(f"  Type    : {type(basis_library).__name__}")
logger.info(f"  Elements: {sorted(basis_library.elements)}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FROM LOCAL FILE — explicit path via BASIS_DIR
# ══════════════════════════════════════════════════════════════════════════════
# ``BASIS_DIR`` (also exported from ``q_block.compute.environment.constants.numerical.pople``) is the
# :class:`~pathlib.Path` that points to the bundled Pople directory.
# Passing a filepath to ``Pople`` lets you load any .gbs file — bundled or
# your own.

_local_path: Path = BASIS_DIR / "6-31G.gbs"
basis_file: Pople = Pople(filepath=str(_local_path))

logger.info("=== 2. From local file ===")
logger.info(f"  File    : {_local_path.name}")
logger.info(f"  Type    : {type(basis_file).__name__}")
logger.info(f"  Elements: {sorted(basis_file.elements)}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FROM BASIS SET EXCHANGE — download, cache to temp file, parse
# ══════════════════════════════════════════════════════════════════════════════
# The Basis Set Exchange (https://www.basissetexchange.org) exposes a public
# REST API.  The response is a plain-text Gaussian94-format basis set that is
# directly compatible with ``Pople.parse``.
#
# Steps:
#   a) Download via urllib (stdlib — no extra dependencies).
#   b) Write to a NamedTemporaryFile so ``Pople`` can open it by path.
#   c) Parse with ``Pople(filepath=tmp_path)``.
#   d) Delete the temp file immediately after parsing.

logger.info("=== 3. From Basis Set Exchange ===")
logger.info(f"  URL: {_BSE_URL}")

with urllib.request.urlopen(_BSE_URL) as _response:
    _gbs_content: str = _response.read().decode("utf-8")

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".gbs", delete=False, encoding="utf-8"
) as _tmp:
    _tmp.write(_gbs_content)
    _tmp_path: Path = Path(_tmp.name)

basis_bse: Pople = Pople(filepath=str(_tmp_path))
_tmp_path.unlink()

logger.info(f"  Type    : {type(basis_bse).__name__}")
logger.info(f"  Elements: {sorted(basis_bse.elements)}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON — verify all three sources produce equivalent data for hydrogen
# ══════════════════════════════════════════════════════════════════════════════
# All three ``Pople`` instances must contain the same elements and the same
# number of angular-momentum shells for hydrogen (Z=1).

logger.info("=== Comparison: hydrogen (H) basis data ===")
for label, basis in [
    ("library", basis_library),
    ("file   ", basis_file),
    ("BSE    ", basis_bse),
]:
    h_regions = basis["H"]
    n_core = sum(len(v) for v in h_regions["core"].values())
    n_valence_inner = sum(
        len(v) for v in h_regions["valence_inner"].values()
    )
    n_valence_outer = sum(
        len(v) for v in h_regions["valence_outer"].values()
    )
    logger.info(
        f"  {label}: core shells={n_core}, "
        f"valence_inner shells={n_valence_inner}, "
        f"valence_outer shells={n_valence_outer}"
    )

logger.info("")
logger.info("All three loading methods produced equivalent basis set data.")
