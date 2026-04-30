"""Common testing utilities for qBlock tests.

This module provides shared fixtures, constants, and helper functions
used across all test modules (integrals, initialization, etc.).
"""

from typing import List, Tuple

import numpy as np
import pytest

from q_block.compute import Molecule
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.solvers.wavefunction.hartree_fock.restricted_hartree_fock import (
    RestrictedHartreeFock,
)
from q_block.compute.solvers.wavefunction.hartree_fock.restricted_open_shell_hartree_fock import (
    RestrictedOpenShellHartreeFock,
)
from q_block.compute.solvers.wavefunction.hartree_fock.unrestricted_hartree_fock import (
    UnrestrictedHartreeFock,
)
from q_block.compute.utilities.mathematics import get_cartesian_components
from q_block.tests.verification.environment.constants import BASIS_STO_3G

# ======================================================================
# Orbital Component Constants
# ======================================================================

# All orbital types: s (l=0), p (l=1), d (l=2)
ALL_ORBITAL_COMPONENTS: List[Tuple[int, int, int]] = (
    get_cartesian_components(0)  # s: [(0,0,0)]
    + get_cartesian_components(1)  # p: [(1,0,0), (0,1,0), (0,0,1)]
    + get_cartesian_components(
        2
    )  # d: [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]
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


def extract_nuclei(
    molecule: Molecule,
) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Extract nuclear positions and charges from a molecule.

    :param molecule: Molecule instance with atoms.
    :type molecule: Molecule
    :returns: List of (Z, (x, y, z)) tuples for each nucleus.
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return [
        (
            atom.atomic_number,
            (atom.coordinates.x, atom.coordinates.y, atom.coordinates.z),
        )
        for atom in molecule.atoms
    ]


# ======================================================================
# Matrix Generation Helpers
# ======================================================================


def make_symmetric(n: int, seed: int) -> np.ndarray:
    """Generate a random symmetric matrix.

    Produces a reproducible n×n symmetric matrix by creating a random
    matrix and adding its transpose.  Useful for constructing test
    Fock-like matrices.

    :param n: Matrix dimension.
    :type n: int
    :param seed: RNG seed for reproducibility.
    :type seed: int
    :returns: Symmetric n×n matrix.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    return A + A.T


def make_overlap(n: int, seed: int) -> np.ndarray:
    """Generate a random positive-definite overlap matrix.

    Constructs a reproducible positive-definite n×n matrix via
    A @ A.T + n * I, which guarantees all eigenvalues are positive.
    Useful for constructing test overlap matrices in generalised
    eigenvalue problems.

    :param n: Matrix dimension.
    :type n: int
    :param seed: RNG seed for reproducibility.
    :type seed: int
    :returns: Positive-definite symmetric n×n matrix.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    return A @ A.T + n * np.eye(n)


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
def h2_nuclei(
    h2_molecule: Molecule,
) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Nuclear positions and charges for H2.

    :param h2_molecule: H2 molecule fixture.
    :type h2_molecule: Molecule
    :returns: List of (Z, (x, y, z)) for each H nucleus.
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return extract_nuclei(h2_molecule)


@pytest.fixture
def water_nuclei(
    water_molecule: Molecule,
) -> List[Tuple[int, Tuple[float, float, float]]]:
    """Nuclear positions and charges for water.

    :param water_molecule: Water molecule fixture.
    :type water_molecule: Molecule
    :returns: List of (Z, (x, y, z)) for each nucleus (O, H, H).
    :rtype: List[Tuple[int, Tuple[float, float, float]]]
    """
    return extract_nuclei(molecule=water_molecule)


@pytest.fixture
def h2_rhf(h2_molecule: Molecule) -> RestrictedHartreeFock:
    """RHF instance for H2 / STO-3G (not yet run).

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    :returns: Uninitialised RHF instance ready for .run().
    :rtype: RestrictedHartreeFock
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    return RestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_electrons=2,
    )


@pytest.fixture
def h2_uhf(h2_molecule: Molecule) -> UnrestrictedHartreeFock:
    """UHF instance for H2 / STO-3G, closed-shell (not yet run).

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    :returns: Uninitialised UHF instance ready for .run().
    :rtype: UnrestrictedHartreeFock
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    return UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_alpha=1,
        n_beta=1,
    )


@pytest.fixture
def water_rhf(water_molecule: Molecule) -> RestrictedHartreeFock:
    """RHF instance for water / STO-3G (not yet run).

    :param water_molecule: Pytest fixture providing a water Molecule with STO-3G basis in Bohr.
    :type water_molecule: Molecule
    :returns: Uninitialised RHF instance ready for .run().
    :rtype: RestrictedHartreeFock
    """
    nuclei = extract_nuclei(molecule=water_molecule)
    cgtos = water_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=water_molecule).energy
    return RestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_electrons=10,
    )


@pytest.fixture
def water_uhf(water_molecule: Molecule) -> UnrestrictedHartreeFock:
    """UHF instance for water / STO-3G, closed-shell (not yet run).

    :param water_molecule: Pytest fixture providing a water Molecule with STO-3G basis in Bohr.
    :type water_molecule: Molecule
    :returns: Uninitialised UHF instance ready for .run().
    :rtype: UnrestrictedHartreeFock
    """
    nuclei = extract_nuclei(molecule=water_molecule)
    cgtos = water_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=water_molecule).energy
    return UnrestrictedHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_alpha=5,
        n_beta=5,
    )


@pytest.fixture
def h2_rohf(h2_molecule: Molecule) -> RestrictedOpenShellHartreeFock:
    """ROHF instance for H2 / STO-3G, closed-shell (n_open=0, not yet run).

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    :returns: Uninitialised ROHF instance ready for .run().
    :rtype: RestrictedOpenShellHartreeFock
    """
    nuclei = extract_nuclei(molecule=h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=h2_molecule).energy
    return RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_closed=1,
        n_open=0,
    )


@pytest.fixture
def water_rohf(water_molecule: Molecule) -> RestrictedOpenShellHartreeFock:
    """ROHF instance for water / STO-3G, closed-shell (n_open=0, not yet run).

    :param water_molecule: Pytest fixture providing a water Molecule with STO-3G basis in Bohr.
    :type water_molecule: Molecule
    :returns: Uninitialised ROHF instance ready for .run().
    :rtype: RestrictedOpenShellHartreeFock
    """
    nuclei = extract_nuclei(molecule=water_molecule)
    cgtos = water_molecule.contracted_gaussian_type_orbitals
    e_nuc = NuclearRepulsionEnergy(molecule=water_molecule).energy
    return RestrictedOpenShellHartreeFock(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_closed=5,
        n_open=0,
    )
