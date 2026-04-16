"""Unit tests for DFT calculation context classes.

Tests cover:
- DensityFunctionalTheory: generic DFT parent class
- RKS: Restricted Closed-Shell KS
- UKS: Unrestricted KS

All tests use pytest with parametrize, no test classes.
"""

from typing import List, Tuple

import pytest

from compute.environment.io.basis_set import Pople
from compute.environment.io.input_data import InputData
from compute.models.initialization import RKS, UKS, DensityFunctionalTheory
from compute.models.molecule import Molecule
from tests.verification.environment.constants import BASIS_6_31G


# ======================================================================
# Helper Functions
# ======================================================================


def _make_molecule(
    atoms_spec: List[Tuple[str, int]],
    basis: Pople = BASIS_6_31G,
    multiplicity: int = 1,
) -> Molecule:
    """Build Molecule from (symbol, charge) pairs.

    :param atoms_spec: List of (symbol, charge) tuples.
    :type atoms_spec: List[Tuple[str, int]]
    :param basis: Basis set for all atoms.
    :type basis: Pople
    :param multiplicity: Spin multiplicity.
    :type multiplicity: int
    :returns: Fully constructed Molecule.
    :rtype: Molecule
    """
    atom_data: List = [
        [sym, 0.0, 0.0, 0.0, basis, q]
        for sym, q in atoms_spec
    ]
    inp: InputData = InputData()
    inp.from_script(atom_data=atom_data, atom_prefix="T")
    return Molecule(input_data=inp, multiplicity=multiplicity)


# ======================================================================
# DensityFunctionalTheory base class
# ======================================================================


def test_dft_base_attributes_water() -> None:
    """DensityFunctionalTheory stores correct attributes for water."""
    water: Molecule = _make_molecule(
        [("O", 0), ("H", 0), ("H", 0)]
    )
    ctx = DensityFunctionalTheory(
        molecule=water, ks_method="RKS"
    )
    assert ctx.molecule is water
    assert ctx.charge == 0
    assert ctx.multiplicity == 1
    assert ctx.n_electrons == 10
    assert ctx.ks_method == "RKS"
    assert ctx.n_basis > 0


def test_dft_base_cgto_lazy_property() -> None:
    """DFT context builds CGTOs lazily on first access."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = DensityFunctionalTheory(
        molecule=h2, ks_method="RKS"
    )
    assert ctx._cgto is None
    cgtos = ctx.cgto
    assert len(cgtos) > 0
    assert ctx._cgto is not None


def test_dft_base_repr() -> None:
    """DensityFunctionalTheory repr contains expected fields."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = DensityFunctionalTheory(
        molecule=h2, ks_method="RKS"
    )
    r = repr(ctx)
    assert "DensityFunctionalTheory" in r
    assert "ks_method=RKS" in r
    assert "n_electrons=2" in r


# ======================================================================
# RKS Tests
# ======================================================================


def test_rks_h2() -> None:
    """RKS attributes are correct for H₂."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = RKS(molecule=h2)
    assert ctx.ks_method == "RKS"
    assert ctx.n_electrons == 2
    assert ctx.n_occ == 1
    assert ctx.n_alpha == 1
    assert ctx.n_beta == 1
    assert ctx.n_basis > 0


def test_rks_water() -> None:
    """RKS attributes are correct for water."""
    water: Molecule = _make_molecule(
        [("O", 0), ("H", 0), ("H", 0)]
    )
    ctx = RKS(molecule=water)
    assert ctx.n_electrons == 10
    assert ctx.n_occ == 5
    assert ctx.n_alpha == 5
    assert ctx.n_beta == 5


def test_rks_odd_electrons_raises() -> None:
    """RKS raises ValueError for odd number of electrons."""
    h: Molecule = _make_molecule(
        [("H", 0)], multiplicity=2
    )
    with pytest.raises(ValueError, match="even number of electrons"):
        RKS(molecule=h)


def test_rks_non_singlet_raises() -> None:
    """RKS raises ValueError for non-singlet multiplicity."""
    h2: Molecule = _make_molecule(
        [("H", 0), ("H", 0)], multiplicity=3
    )
    with pytest.raises(ValueError, match="singlet multiplicity"):
        RKS(molecule=h2)


def test_rks_repr() -> None:
    """RKS repr contains expected fields."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = RKS(molecule=h2)
    r = repr(ctx)
    assert "RKS" in r
    assert "n_occ=1" in r


def test_rks_inherits_dft() -> None:
    """RKS inherits from DensityFunctionalTheory."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = RKS(molecule=h2)
    assert isinstance(ctx, DensityFunctionalTheory)


# ======================================================================
# UKS Tests
# ======================================================================


def test_uks_h_atom() -> None:
    """UKS attributes are correct for a single H atom (doublet)."""
    h: Molecule = _make_molecule(
        [("H", 0)], multiplicity=2
    )
    ctx = UKS(molecule=h)
    assert ctx.ks_method == "UKS"
    assert ctx.n_electrons == 1
    assert ctx.n_alpha == 1
    assert ctx.n_beta == 0


def test_uks_h2() -> None:
    """UKS attributes are correct for H₂ (singlet)."""
    h2: Molecule = _make_molecule([("H", 0), ("H", 0)])
    ctx = UKS(molecule=h2)
    assert ctx.n_electrons == 2
    assert ctx.n_alpha == 1
    assert ctx.n_beta == 1


def test_uks_triplet_o2() -> None:
    """UKS computes correct alpha/beta for O₂ triplet."""
    o2: Molecule = _make_molecule(
        [("O", 0), ("O", 0)], multiplicity=3
    )
    ctx = UKS(molecule=o2)
    assert ctx.n_electrons == 16
    assert ctx.n_alpha == 9
    assert ctx.n_beta == 7


def test_uks_repr() -> None:
    """UKS repr contains expected fields."""
    h: Molecule = _make_molecule(
        [("H", 0)], multiplicity=2
    )
    ctx = UKS(molecule=h)
    r = repr(ctx)
    assert "UKS" in r
    assert "n_alpha=1" in r
    assert "n_beta=0" in r


def test_uks_inherits_dft() -> None:
    """UKS inherits from DensityFunctionalTheory."""
    h: Molecule = _make_molecule(
        [("H", 0)], multiplicity=2
    )
    ctx = UKS(molecule=h)
    assert isinstance(ctx, DensityFunctionalTheory)
