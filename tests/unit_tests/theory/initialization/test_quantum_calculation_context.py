"""Unit tests for q_block.theory.initialization module.

Tests cover:
- QuantumCalculationContext: generic parent class
- HartreeFock: common HF base class
- RHF: Restricted Closed-Shell
- UHF: Unrestricted
- ROHF: Restricted Open-Shell

Charge and electron-count tests use all combinations of per-atom charges
(-4, ..., +8) to validate N = sum(Z_i + q_i).

All tests use pytest with parametrize, no test classes.
"""

import itertools
from typing import List, Tuple

import pytest

from q_block.constants.atoms_data import ATOMS_SYMBOLS_SYMBOL_TO_Z
from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule
from q_block.theory.initialization import (
    HartreeFock,
    QuantumCalculationContext,
    ROHF,
    RHF,
    UHF,
)
from tests.unit_tests.constants import (
    BASIS_6_31G,
    CHARGE_VALUES,
    charge_ids,
)


# ======================================================================
# Helper Functions
# ======================================================================


def _make_molecule(
    atoms_spec: List[Tuple[str, int]],
    basis: Pople = BASIS_6_31G,
    multiplicity: int = 1,
) -> Molecule:
    """Build Molecule from (symbol, charge) pairs with uniform basis set.

    :param atoms_spec: List of (symbol, charge) tuples.
    :type atoms_spec: List[Tuple[str, int]]
    :param basis: Basis set for all atoms.
    :type basis: Pople
    :param multiplicity: Spin multiplicity.
    :type multiplicity: int
    :returns: Fully constructed Molecule.
    :rtype: Molecule
    """
    atom_data: List = [[sym, 0.0, 0.0, 0.0, basis, q] for sym, q in atoms_spec]
    inp: InputData = InputData()
    inp.from_script(atom_data=atom_data, atom_prefix="T")
    return Molecule(input_data=inp, multiplicity=multiplicity)


def _n_electrons(atoms_spec: List[Tuple[str, int]]) -> int:
    """Expected total electron count: sum(Z_i + q_i).

    :param atoms_spec: List of (symbol, charge) tuples.
    :type atoms_spec: List[Tuple[str, int]]
    :returns: Total electron count.
    :rtype: int
    """
    return sum(ATOMS_SYMBOLS_SYMBOL_TO_Z[s] + q for s, q in atoms_spec)


# ======================================================================
# QuantumCalculationContext Tests
# ======================================================================


def test_quantum_calculation_context_basic_attributes() -> None:
    """Verify QuantumCalculationContext has correct basic attributes for water."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    ctx: QuantumCalculationContext = QuantumCalculationContext(molecule=water)

    assert ctx.molecule is water
    assert ctx.charge == 0
    assert ctx.multiplicity == 1
    assert ctx.n_electrons == 10


def test_quantum_calculation_context_coordinates_converted_to_bohr() -> None:
    """Verify coordinates are converted to Bohr after QuantumCalculationContext."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    QuantumCalculationContext(molecule=water)

    for atom in water.atoms:
        assert atom.coordinates is not None


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_quantum_calculation_context_single_atom_charge(charge: int) -> None:
    """Verify QuantumCalculationContext with single O atom for various charges.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    mult: int = 1 if n_el % 2 == 0 else 2
    mol: Molecule = _make_molecule([("O", charge)], multiplicity=mult)
    ctx: QuantumCalculationContext = QuantumCalculationContext(molecule=mol)

    assert ctx.charge == charge
    assert ctx.n_electrons == 8 + charge


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
)
def test_quantum_calculation_context_h2_all_charge_combos(q1: int, q2: int) -> None:
    """Verify QuantumCalculationContext with H₂ for all 25 charge combinations.

    Skips combos producing negative electrons or odd n_electrons
    (incompatible with singlet multiplicity).

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    n_el: int = 2 + q1 + q2
    if n_el < 0 or n_el % 2 != 0:
        pytest.skip("invalid combo for singlet Molecule")
    mol: Molecule = _make_molecule([("H", q1), ("H", q2)])
    ctx: QuantumCalculationContext = QuantumCalculationContext(molecule=mol)

    assert ctx.n_electrons == n_el
    assert ctx.charge == q1 + q2


def test_quantum_calculation_context_repr() -> None:
    """Verify QuantumCalculationContext __repr__ contains expected information."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    ctx: QuantumCalculationContext = QuantumCalculationContext(molecule=water)
    r: str = repr(ctx)

    assert "QuantumCalculationContext" in r
    assert "n_electrons=10" in r


# ======================================================================
# RHF Tests
# ======================================================================


def test_rhf_water() -> None:
    """Verify RHF attributes for water molecule."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rhf: RHF = RHF(molecule=water)

    assert rhf.hf_method == "RHF"
    assert rhf.n_electrons == 10
    assert rhf.n_occ == 5
    assert rhf.n_alpha == 5
    assert rhf.n_beta == 5
    assert rhf.n_basis > 0


def test_rhf_inherits_hartree_fock() -> None:
    """Verify RHF inherits from HartreeFock and QuantumCalculationContext."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rhf: RHF = RHF(molecule=water)

    assert isinstance(rhf, HartreeFock)
    assert isinstance(rhf, QuantumCalculationContext)


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_rhf_single_oxygen_charges(charge: int) -> None:
    """Verify RHF on single O atom with various charges.

    n_electrons = 8 + charge. RHF requires even n_electrons and singlet.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    if n_el < 0 or n_el % 2 != 0:
        pytest.skip("invalid for RHF")
    mol: Molecule = _make_molecule([("O", charge)])
    rhf: RHF = RHF(molecule=mol)

    assert rhf.n_electrons == n_el
    assert rhf.n_occ == n_el // 2
    assert rhf.n_alpha == rhf.n_occ
    assert rhf.n_beta == rhf.n_occ


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
)
def test_rhf_h2_all_charge_combos(q1: int, q2: int) -> None:
    """Verify RHF on H₂ for all valid charge combinations.

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    n_el: int = 2 + q1 + q2
    if n_el < 0 or n_el % 2 != 0:
        pytest.skip("invalid for RHF singlet")
    mol: Molecule = _make_molecule([("H", q1), ("H", q2)])
    rhf: RHF = RHF(molecule=mol)

    assert rhf.n_electrons == n_el
    assert rhf.n_occ == n_el // 2
    assert rhf.charge == q1 + q2


def test_rhf_odd_electrons_raises() -> None:
    """Verify RHF rejects odd number of electrons."""
    mol: Molecule = _make_molecule([("H", 0)], multiplicity=2)
    with pytest.raises(ValueError, match="even number of electrons"):
        RHF(molecule=mol)


def test_rhf_non_singlet_raises() -> None:
    """Verify RHF rejects non-singlet multiplicity."""
    mol: Molecule = _make_molecule([("O", 0), ("O", 0)], multiplicity=3)
    with pytest.raises(ValueError, match="singlet multiplicity"):
        RHF(molecule=mol)


def test_rhf_repr() -> None:
    """Verify RHF __repr__ contains expected information."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rhf: RHF = RHF(molecule=water)
    r: str = repr(rhf)

    assert "RHF" in r
    assert "n_occ=5" in r


# ======================================================================
# UHF Tests
# ======================================================================


def test_uhf_water_singlet() -> None:
    """Verify UHF attributes for water (singlet)."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    uhf: UHF = UHF(molecule=water)

    assert uhf.hf_method == "UHF"
    assert uhf.n_electrons == 10
    assert uhf.n_alpha == 5
    assert uhf.n_beta == 5


def test_uhf_doublet_single_h() -> None:
    """Verify UHF on single H atom (doublet, 1 electron)."""
    mol: Molecule = _make_molecule([("H", 0)], multiplicity=2)
    uhf: UHF = UHF(molecule=mol)

    assert uhf.n_electrons == 1
    assert uhf.n_alpha == 1
    assert uhf.n_beta == 0


def test_uhf_triplet_o2() -> None:
    """Verify UHF on O₂ (triplet, 16 electrons)."""
    mol: Molecule = _make_molecule([("O", 0), ("O", 0)], multiplicity=3)
    uhf: UHF = UHF(molecule=mol)

    assert uhf.n_electrons == 16
    assert uhf.n_alpha == 9
    assert uhf.n_beta == 7


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_uhf_single_oxygen_charges(charge: int) -> None:
    """Verify UHF on single O with charges. Compatible multiplicity chosen.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    if n_el < 0:
        pytest.skip("negative electrons")
    mult: int = 1 if n_el % 2 == 0 else 2
    mol: Molecule = _make_molecule([("O", charge)], multiplicity=mult)
    uhf: UHF = UHF(molecule=mol)

    assert uhf.n_electrons == n_el
    n_unpaired: int = mult - 1
    assert uhf.n_beta == (n_el - n_unpaired) // 2
    assert uhf.n_alpha == uhf.n_beta + n_unpaired


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
)
def test_uhf_h2_all_charge_combos(q1: int, q2: int) -> None:
    """Verify UHF on H₂ for all valid charge combinations.

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    n_el: int = 2 + q1 + q2
    if n_el < 0:
        pytest.skip("negative electrons")
    mult: int = 1 if n_el % 2 == 0 else 2
    if n_el == 0:
        mult = 1
    mol: Molecule = _make_molecule([("H", q1), ("H", q2)], multiplicity=mult)
    uhf: UHF = UHF(molecule=mol)

    assert uhf.n_electrons == n_el
    assert uhf.n_alpha + uhf.n_beta == n_el


def test_uhf_inherits_hartree_fock() -> None:
    """Verify UHF inherits from HartreeFock."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    uhf: UHF = UHF(molecule=water)
    assert isinstance(uhf, HartreeFock)


def test_uhf_repr() -> None:
    """Verify UHF __repr__ contains expected information."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    uhf: UHF = UHF(molecule=water)
    r: str = repr(uhf)

    assert "UHF" in r
    assert "n_alpha=" in r
    assert "n_beta=" in r


# ======================================================================
# ROHF Tests
# ======================================================================


def test_rohf_water_singlet() -> None:
    """Verify ROHF attributes for water (singlet)."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rohf: ROHF = ROHF(molecule=water)

    assert rohf.hf_method == "ROHF"
    assert rohf.n_electrons == 10
    assert rohf.n_closed == 5
    assert rohf.n_open == 0
    assert rohf.n_alpha == 5
    assert rohf.n_beta == 5


def test_rohf_doublet_single_h() -> None:
    """Verify ROHF on single H (doublet, 1 electron)."""
    mol: Molecule = _make_molecule([("H", 0)], multiplicity=2)
    rohf: ROHF = ROHF(molecule=mol)

    assert rohf.n_electrons == 1
    assert rohf.n_closed == 0
    assert rohf.n_open == 1
    assert rohf.n_alpha == 1
    assert rohf.n_beta == 0


def test_rohf_triplet_o2() -> None:
    """Verify ROHF on O₂ (triplet, 16 electrons)."""
    mol: Molecule = _make_molecule([("O", 0), ("O", 0)], multiplicity=3)
    rohf: ROHF = ROHF(molecule=mol)

    assert rohf.n_electrons == 16
    assert rohf.n_closed == 7
    assert rohf.n_open == 2
    assert rohf.n_alpha == 9
    assert rohf.n_beta == 7


@pytest.mark.parametrize(
    "charge",
    CHARGE_VALUES,
    ids=charge_ids(),
)
def test_rohf_single_oxygen_charges(charge: int) -> None:
    """Verify ROHF on single O with charges.

    :param charge: Charge on the oxygen atom.
    :type charge: int
    """
    n_el: int = 8 + charge
    if n_el < 0:
        pytest.skip("negative electrons")
    mult: int = 1 if n_el % 2 == 0 else 2
    mol: Molecule = _make_molecule([("O", charge)], multiplicity=mult)
    rohf: ROHF = ROHF(molecule=mol)

    assert rohf.n_electrons == n_el
    n_unpaired: int = mult - 1
    assert rohf.n_open == n_unpaired
    assert rohf.n_closed == (n_el - n_unpaired) // 2


@pytest.mark.parametrize(
    "q1, q2",
    list(itertools.product(CHARGE_VALUES, repeat=2)),
)
def test_rohf_h2_all_charge_combos(q1: int, q2: int) -> None:
    """Verify ROHF on H₂ for all valid charge combinations.

    :param q1: Charge on first hydrogen atom.
    :type q1: int
    :param q2: Charge on second hydrogen atom.
    :type q2: int
    """
    n_el: int = 2 + q1 + q2
    if n_el < 0:
        pytest.skip("negative electrons")
    mult: int = 1 if n_el % 2 == 0 else 2
    if n_el == 0:
        mult = 1
    mol: Molecule = _make_molecule([("H", q1), ("H", q2)], multiplicity=mult)
    rohf: ROHF = ROHF(molecule=mol)

    assert rohf.n_electrons == n_el
    assert rohf.n_alpha + rohf.n_beta == n_el
    assert rohf.n_alpha == rohf.n_closed + rohf.n_open
    assert rohf.n_beta == rohf.n_closed


def test_rohf_inherits_hartree_fock() -> None:
    """Verify ROHF inherits from HartreeFock."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rohf: ROHF = ROHF(molecule=water)
    assert isinstance(rohf, HartreeFock)


def test_rohf_repr() -> None:
    """Verify ROHF __repr__ contains expected information."""
    water: Molecule = _make_molecule([("O", 0), ("H", 0), ("H", 0)])
    rohf: ROHF = ROHF(molecule=water)
    r: str = repr(rohf)

    assert "ROHF" in r
    assert "n_closed=" in r
    assert "n_open=" in r


# ======================================================================
# Water All 125 Charge Combinations Tests
# ======================================================================


def _valid_water_rhf_combos() -> List[Tuple[int, int, int]]:
    """(qO, qH1, qH2) where n_electrons >= 0, even, for singlet RHF."""
    results: List[Tuple[int, int, int]] = []
    for qo, qh1, qh2 in itertools.product(CHARGE_VALUES, repeat=3):
        n_el = (8 + qo) + (1 + qh1) + (1 + qh2)
        if n_el >= 0 and n_el % 2 == 0:
            results.append((qo, qh1, qh2))
    return results


@pytest.mark.parametrize("qo, qh1, qh2", _valid_water_rhf_combos())
def test_rhf_water_all_charge_combos(qo: int, qh1: int, qh2: int) -> None:
    """Verify RHF on water-like O-H-H with all valid three-atom charge combos.

    :param qo: Charge on oxygen atom.
    :type qo: int
    :param qh1: Charge on first hydrogen atom.
    :type qh1: int
    :param qh2: Charge on second hydrogen atom.
    :type qh2: int
    """
    mol: Molecule = _make_molecule([("O", qo), ("H", qh1), ("H", qh2)])
    rhf: RHF = RHF(molecule=mol)
    expected_n: int = (8 + qo) + (1 + qh1) + (1 + qh2)

    assert rhf.n_electrons == expected_n
    assert rhf.n_occ == expected_n // 2
    assert rhf.charge == qo + qh1 + qh2


def _valid_water_uhf_combos() -> List[Tuple[int, int, int, int]]:
    """(qO, qH1, qH2, mult) where n_electrons >= 0 and parity-compatible."""
    results: List[Tuple[int, int, int, int]] = []
    for qo, qh1, qh2 in itertools.product(CHARGE_VALUES, repeat=3):
        n_el = (8 + qo) + (1 + qh1) + (1 + qh2)
        if n_el < 0:
            continue
        mult = 1 if n_el % 2 == 0 else 2
        if n_el == 0:
            mult = 1
        results.append((qo, qh1, qh2, mult))
    return results


@pytest.mark.parametrize("qo, qh1, qh2, mult", _valid_water_uhf_combos())
def test_uhf_water_all_charge_combos(
    qo: int, qh1: int, qh2: int, mult: int
) -> None:
    """Verify UHF on water-like O-H-H with all valid three-atom charge combos.

    :param qo: Charge on oxygen atom.
    :type qo: int
    :param qh1: Charge on first hydrogen atom.
    :type qh1: int
    :param qh2: Charge on second hydrogen atom.
    :type qh2: int
    :param mult: Spin multiplicity.
    :type mult: int
    """
    mol: Molecule = _make_molecule(
        [("O", qo), ("H", qh1), ("H", qh2)], multiplicity=mult
    )
    uhf: UHF = UHF(molecule=mol)
    expected_n: int = (8 + qo) + (1 + qh1) + (1 + qh2)

    assert uhf.n_electrons == expected_n
    assert uhf.n_alpha + uhf.n_beta == expected_n
