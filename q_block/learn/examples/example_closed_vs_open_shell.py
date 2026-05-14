"""
Example: Closed-Shell vs Open-Shell SCF Comparison
====================================================

Compares all applicable HF and DFT methods on two molecules drawn
directly from the validation dataset:

    Closed-shell  H₂O  (singlet, multiplicity = 1)  →  RHF, RKS-B3LYP
    Open-shell    NO   (doublet, multiplicity = 2)   →  UHF, ROHF,
                                                         UKS-B3LYP, ROKS-B3LYP

All methods share the 6-31G basis set.  Geometries are taken from
``q_block/tests/validation/validation_data.py`` (MOLECULES dict).
"""

from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.models.initialization import RHF, ROHF, RKS, ROKS, UHF, UKS
from q_block.compute.models.molecule import Molecule
from q_block.compute.environment.constants.numerical.pople import G631
from q_block.compute.solvers.electronic_density.functionals import B3LYP
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
    RestrictedOpenShellKohnSham,
    UnrestrictedKohnSham,
)
from q_block.compute.solvers.wavefunction.hartree_fock import (
    RestrictedHartreeFock,
    RestrictedOpenShellHartreeFock,
    UnrestrictedHartreeFock,
)

logger = setup_logging(__name__)


def build_cgtos(
    geometry: list,
    multiplicity: int,
) -> tuple:
    r"""Build contracted GTOs from a geometry list.

    Constructs an
    :class:`~q_block.compute.environment.io.input_data.InputData`
    from *geometry*, assembles a
    :class:`~q_block.compute.models.molecule.Molecule`, converts
    coordinates to Bohr, and expands each atom into contracted
    Gaussian-type orbitals using the module-level basis set
    :data:`G631` (6-31G).

    :param geometry: List of atom dictionaries, each with keys
        ``"symbol"`` (:class:`str`), ``"x"``, ``"y"``, ``"z"``
        (:class:`float`, in Ångström).
    :type geometry: list
    :param multiplicity: Spin multiplicity :math:`2S + 1`.
    :type multiplicity: int

    :returns: Pair ``(cgtos, molecule)`` where *cgtos* is the flat
        list of
        :class:`~q_block.compute.models.basis_functions.ContractedGaussianTypeOrbital`
        shells and *molecule* is the fully-initialised
        :class:`~q_block.compute.models.molecule.Molecule`.
    :rtype: tuple
    """
    inp = InputData()
    inp.from_script(
        atom_data=[
            [a["symbol"], a["x"], a["y"], a["z"], G631]
            for a in geometry
        ]
    )
    mol = Molecule(input_data=inp, multiplicity=multiplicity)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol.contracted_gaussian_type_orbitals, mol


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIES (source: NIST CCCBDB experimental)
# ══════════════════════════════════════════════════════════════════════════════

# Closed shell
H2O_GEOMETRY = [
    {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.0000},
    {"symbol": "H", "x": 0.7572, "y": 0.0000, "z": 0.5860},
    {"symbol": "H", "x": -0.7572, "y": 0.0000, "z": 0.5860},
]

# Open shell
NO_GEOMETRY = [
    {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5754},
    {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.5754},
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLOSED-SHELL: H₂O  (singlet)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("CLOSED-SHELL: H₂O  (singlet, multiplicity = 1)")
logger.info("=" * 70)

# ── Build H₂O once with 6-31G (shared by RHF and RKS) ─────────────────────
cgtos_h2o, mol_h2o = build_cgtos(H2O_GEOMETRY, 1)
ctx_h2o_rhf = RHF(molecule=mol_h2o)

logger.info("RHF – H₂O / 6-31G")
rhf_h2o = RestrictedHartreeFock(
    cgtos=ctx_h2o_rhf.cgto,
    n_electrons=ctx_h2o_rhf.n_electrons,
).run()
logger.info(f"  Converged  : {rhf_h2o.converged}")
logger.info(f"  Iterations : {rhf_h2o.n_iterations}")
logger.info(f"  E_total    : {rhf_h2o.e_total:.10f} Hartree\n")

# ── RKS-B3LYP / 6-31G ───────────────────────────────────────────────────────
ctx_h2o_rks = RKS(molecule=mol_h2o)

logger.info("RKS-B3LYP – H₂O / 6-31G")
rks_h2o = RestrictedKohnSham(
    cgtos=ctx_h2o_rks.cgto,
    functional=B3LYP(),
    n_electrons=ctx_h2o_rks.n_electrons,
).run()
logger.info(f"  Converged  : {rks_h2o.converged}")
logger.info(f"  Iterations : {rks_h2o.n_iterations}")
logger.info(f"  E_total    : {rks_h2o.e_total:.10f} Hartree\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. OPEN-SHELL: NO  (doublet, multiplicity = 2)
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("OPEN-SHELL: NO  (doublet, multiplicity = 2)")
logger.info("=" * 70)

# ── Build NO once with 6-31G (shared by UHF, ROHF, UKS, ROKS) ─────────────
cgtos_no, mol_no = build_cgtos(NO_GEOMETRY, 2)
ctx_no_uhf = UHF(molecule=mol_no)

logger.info("UHF – NO / 6-31G")
uhf_no = UnrestrictedHartreeFock(
    cgtos=ctx_no_uhf.cgto,
    n_alpha=ctx_no_uhf.n_alpha,
    n_beta=ctx_no_uhf.n_beta,
).run()
logger.info(f"  Converged  : {uhf_no.converged}")
logger.info(f"  Iterations : {uhf_no.n_iterations}")
logger.info(f"  E_total    : {uhf_no.e_total:.10f} Hartree\n")

# ── ROHF / 6-31G ────────────────────────────────────────────────────────────
ctx_no_rohf = ROHF(molecule=mol_no)

logger.info("ROHF – NO / 6-31G")
rohf_no = RestrictedOpenShellHartreeFock(
    cgtos=ctx_no_rohf.cgto,
    n_closed=ctx_no_rohf.n_closed,
    n_open=ctx_no_rohf.n_open,
).run()
logger.info(f"  Converged  : {rohf_no.converged}")
logger.info(f"  Iterations : {rohf_no.n_iterations}")
logger.info(f"  E_total    : {rohf_no.e_total:.10f} Hartree\n")

# ── UKS-B3LYP / 6-31G ───────────────────────────────────────────────────────
ctx_no_uks = UKS(molecule=mol_no)

logger.info("UKS-B3LYP – NO / 6-31G")
uks_no = UnrestrictedKohnSham(
    cgtos=ctx_no_uks.cgto,
    functional=B3LYP(),
    n_alpha=ctx_no_uks.n_alpha,
    n_beta=ctx_no_uks.n_beta,
).run()
logger.info(f"  Converged  : {uks_no.converged}")
logger.info(f"  Iterations : {uks_no.n_iterations}")
logger.info(f"  E_total    : {uks_no.e_total:.10f} Hartree\n")

# ── ROKS-B3LYP / 6-31G ──────────────────────────────────────────────────────
ctx_no_roks = ROKS(molecule=mol_no)

logger.info("ROKS-B3LYP – NO / 6-31G")
roks_no = RestrictedOpenShellKohnSham(
    cgtos=ctx_no_roks.cgto,
    functional=B3LYP(),
    n_closed=ctx_no_roks.n_closed,
    n_open=ctx_no_roks.n_open,
).run()
logger.info(f"  Converged  : {roks_no.converged}")
logger.info(f"  Iterations : {roks_no.n_iterations}")
logger.info(f"  E_total    : {roks_no.e_total:.10f} Hartree\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("Summary: total energies (Hartree)")
logger.info("=" * 70)
logger.info("Closed-shell  H₂O")
logger.info(f"  RHF        : {rhf_h2o.e_total:.10f}")
logger.info(f"  RKS-B3LYP  : {rks_h2o.e_total:.10f}")
logger.info("")
logger.info("Open-shell    NO")
logger.info(f"  UHF        : {uhf_no.e_total:.10f}")
logger.info(f"  ROHF       : {rohf_no.e_total:.10f}")
logger.info(f"  UKS-B3LYP  : {uks_no.e_total:.10f}")
logger.info(f"  ROKS-B3LYP : {roks_no.e_total:.10f}")
logger.info("")
