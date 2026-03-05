"""Common testing utilities for qBlock tests.

This module provides shared fixtures, constants, and helper functions
used across all test modules (integrals, initialization, etc.).
"""

from typing import List, Tuple

import pytest

from q_block import Molecule
from q_block.io.input_data import InputData
from q_block.theory.utils import get_cartesian_components
from tests.constants import BASIS_STO_3G


# ======================================================================
# Orbital Component Constants
# ======================================================================

# All orbital types: s (l=0), p (l=1), d (l=2)
ALL_ORBITAL_COMPONENTS: List[Tuple[int, int, int]] = (
    get_cartesian_components(0)  # s: [(0,0,0)]
    + get_cartesian_components(1)  # p: [(1,0,0), (0,1,0), (0,0,1)]
    + get_cartesian_components(2)  # d: [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]
)
"""All Cartesian orbital components for s, p, d shells."""

# Orbital labels for test IDs
ORBITAL_LABELS = {
    (0, 0, 0): "s",
    (1, 0, 0): "px",
    (0, 1, 0): "py",
    (0, 0, 1): "pz",
    (2, 0, 0): "dxx",
    (1, 1, 0): "dxy",
    (1, 0, 1): "dxz",
    (0, 2, 0): "dyy",
    (0, 1, 1): "dyz",
    (0, 0, 2): "dzz",
}
"""Mapping from (lx, ly, lz) to human-readable orbital labels."""


# ======================================================================
# Helper Functions
# ======================================================================


def orbital_id(component: Tuple[int, int, int]) -> str:
    """Generate test ID for orbital component.

    :param component: Tuple of (lx, ly, lz) angular momentum indices.
    :type component: Tuple[int, int, int]
    :returns: Human-readable orbital label (e.g., 's', 'px', 'dxy').
    :rtype: str
    """
    return ORBITAL_LABELS.get(component, f"l{sum(component)}")


def get_orbital_ids(components: List[Tuple[int, int, int]] = None) -> List[str]:
    """Generate list of test IDs for orbital components.

    :param components: List of (lx, ly, lz) tuples. Defaults to ALL_ORBITAL_COMPONENTS.
    :type components: List[Tuple[int, int, int]], optional
    :returns: List of human-readable orbital labels.
    :rtype: List[str]
    """
    if components is None:
        components = ALL_ORBITAL_COMPONENTS
    return [orbital_id(c) for c in components]


def extract_nuclei(molecule: Molecule) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Extract nuclear positions and charges from a molecule.

    :param molecule: Molecule instance with atoms.
    :type molecule: Molecule
    :returns: List of (Z, (x, y, z)) tuples for each nucleus.
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return [
        (atom.atomic_number, (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z))
        for atom in molecule.atoms
    ]


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def h2_molecule() -> Molecule:
    """H2 molecule with STO-3G basis, converted to Bohr."""
    inp: InputData = InputData()
    inp.from_script(
        atom_data=[
            ["H", 0.0, 0.0, 0.0, BASIS_STO_3G],
            ["H", 0.0, 0.0, 0.74, BASIS_STO_3G],  # ~0.74 Å bond length
        ]
    )
    mol: Molecule = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol


@pytest.fixture
def water_molecule() -> Molecule:
    """Water molecule with STO-3G basis, converted to Bohr."""
    inp: InputData = InputData()
    inp.from_script(
        atom_data=[
            ["O", 0.0, 0.0, 0.1173, BASIS_STO_3G],
            ["H", 0.0, 0.7572, -0.4692, BASIS_STO_3G],
            ["H", 0.0, -0.7572, -0.4692, BASIS_STO_3G],
        ]
    )
    mol: Molecule = Molecule(input_data=inp)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    return mol


@pytest.fixture
def h2_nuclei(h2_molecule: Molecule) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Nuclear positions and charges for H2.

    :param h2_molecule: H2 molecule fixture.
    :type h2_molecule: Molecule
    :returns: List of (Z, (x, y, z)) for each H nucleus.
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return extract_nuclei(h2_molecule)


@pytest.fixture
def water_nuclei(water_molecule: Molecule) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Nuclear positions and charges for water.

    :param water_molecule: Water molecule fixture.
    :type water_molecule: Molecule
    :returns: List of (Z, (x, y, z)) for each nucleus (O, H, H).
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return extract_nuclei(water_molecule)
