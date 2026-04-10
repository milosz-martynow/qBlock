"""Unit tests for q_block.io.basis_set module.

Tests cover:
- BasisSet generic behaviour (lookup, contains, repr, elements)
- Region structure (core / valence_inner / valence_outer)
- Angular momentum and BasisSetParameters structure
- Parametrized over both STO and GTO (split-valence) Pople basis sets

All tests use pytest with parametrize, no test classes.
"""

from typing import List

import pytest

from q_block.environment.io.basis_set import BasisSet
from tests.unit_tests.environment.constants import (
    BASIS_3_21G,
    BASIS_6_31G,
    BASIS_6_311G,
    BASIS_6_311PP_GSS,
    BASIS_STO_3G,
    BASIS_STO_6G,
)

# ======================================================================
# Fixtures: all pre-loaded basis sets grouped for parametrize
# ======================================================================

ALL_BASIS_SETS: List[BasisSet] = [
    BASIS_STO_3G,
    BASIS_STO_6G,
    BASIS_3_21G,
    BASIS_6_31G,
    BASIS_6_311G,
    BASIS_6_311PP_GSS,
]
"""Every available basis set instance — extend this list when new
families or files are added."""

_ALL_IDS: List[str] = [
    "STO-3G",
    "STO-6G",
    "3-21G",
    "6-31G",
    "6-311G",
    "6-311++Gss",
]


# ------------------------------------------------------------------
# __contains__ — symbol and atomic number
# ------------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_contains_hydrogen_by_symbol(basis: BasisSet) -> None:
    """Verify every basis set contains hydrogen via symbol lookup."""
    assert "H" in basis


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_contains_hydrogen_by_atomic_number(
    basis: BasisSet,
) -> None:
    """Verify every basis set contains hydrogen via atomic number."""
    assert 1 in basis


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_not_contains_unknown_symbol(basis: BasisSet) -> None:
    """Verify unknown element symbol returns False."""
    assert "Xx" not in basis


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_not_contains_unknown_atomic_number(
    basis: BasisSet,
) -> None:
    """Verify unknown atomic number returns False."""
    assert 999 not in basis


# ------------------------------------------------------------------
# __getitem__ — symbol and atomic number
# ------------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_getitem_by_symbol(basis: BasisSet) -> None:
    """Verify __getitem__ with a symbol returns a regions dict."""
    regions = basis["H"]
    assert isinstance(regions, dict)
    assert "core" in regions


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_getitem_by_atomic_number(basis: BasisSet) -> None:
    """Verify __getitem__ with an int resolves to the same regions."""
    assert basis[1] == basis["H"]


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_getitem_missing_raises(basis: BasisSet) -> None:
    """Verify KeyError for a missing element."""
    with pytest.raises(KeyError):
        _ = basis["Xx"]


# ------------------------------------------------------------------
# elements property
# ------------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_elements_returns_list(basis: BasisSet) -> None:
    """Verify elements property is a list of strings."""
    elements = basis.elements
    assert isinstance(elements, list)
    assert all(isinstance(e, str) for e in elements)
    assert len(elements) > 0


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_elements_contains_hydrogen(basis: BasisSet) -> None:
    """Every Pople basis set should list hydrogen."""
    assert "H" in basis.elements


# ------------------------------------------------------------------
# __repr__
# ------------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_repr_contains_class_name(basis: BasisSet) -> None:
    """Verify repr starts with the concrete class name."""
    r = repr(basis)
    assert r.startswith(basis.__class__.__name__)


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_repr_contains_filepath(basis: BasisSet) -> None:
    """Verify repr includes the filepath."""
    assert repr(basis.filepath) in repr(basis)


# ------------------------------------------------------------------
# Region structure
# ------------------------------------------------------------------

REGION_NAMES = ("core", "valence_inner", "valence_outer")


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_regions_are_valid_keys(basis: BasisSet) -> None:
    """All region keys in data must belong to the known set."""
    for element in basis.elements:
        regions = basis[element]
        for key in regions:
            assert key in REGION_NAMES


@pytest.mark.parametrize(
    "basis, symbol",
    [
        (BASIS_STO_3G, "H"),
        (BASIS_STO_3G, "C"),
        (BASIS_STO_3G, "N"),
        (BASIS_STO_3G, "O"),
        (BASIS_STO_3G, "Zr"),
        (BASIS_STO_3G, "He"),
        (BASIS_STO_6G, "H"),
        (BASIS_STO_6G, "C"),
        (BASIS_3_21G, "H"),
        (BASIS_3_21G, "C"),
        (BASIS_6_31G, "H"),
        (BASIS_6_31G, "C"),
        (BASIS_6_311G, "H"),
        (BASIS_6_311G, "C"),
        (BASIS_6_311PP_GSS, "H"),
        (BASIS_6_311PP_GSS, "C"),
    ],
    ids=[
        "STO-3G_Z1_H",
        "STO-3G_Z6_C",
        "STO-3G_Z7_N",
        "STO-3G_Z8_O",
        "STO-3G_Z40_Zr",
        "STO-3G_Z2_He",
        "STO-6G_Z1_H",
        "STO-6G_Z6_C",
        "3-21G_Z1_H",
        "3-21G_Z6_C",
        "6-31G_Z1_H",
        "6-31G_Z6_C",
        "6-311G_Z1_H",
        "6-311G_Z6_C",
        "6-311++Gss_Z1_H",
        "6-311++Gss_Z6_C",
    ],
)
def test_basis_set_has_core_region(basis: BasisSet, symbol: str) -> None:
    """Verify the basis set has a non-empty core region for the element.

    :param basis: Pre-loaded basis set instance.
    :type basis: BasisSet
    :param symbol: Element symbol to check.
    :type symbol: str
    """
    regions = basis[symbol]
    assert "core" in regions
    assert len(regions["core"]) > 0


# ------------------------------------------------------------------
# Angular momentum and BasisSetParameters structure
# ------------------------------------------------------------------


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_angular_momentum_keys_are_ints(
    basis: BasisSet,
) -> None:
    """Angular momentum keys within each region must be integers."""
    for element in basis.elements:
        for region in basis[element].values():
            for l_key in region:
                assert isinstance(l_key, int)


@pytest.mark.parametrize("basis", ALL_BASIS_SETS, ids=_ALL_IDS)
def test_basis_set_parameters_structure(basis: BasisSet) -> None:
    """Each shell must have 'exponents' and 'coefficients' lists."""
    for element in basis.elements:
        for region in basis[element].values():
            for shells in region.values():
                for params in shells:
                    assert "exponents" in params
                    assert "coefficients" in params
                    assert isinstance(params["exponents"], list)
                    assert isinstance(params["coefficients"], list)
                    assert len(params["exponents"]) == len(params["coefficients"])
                    assert len(params["exponents"]) > 0
