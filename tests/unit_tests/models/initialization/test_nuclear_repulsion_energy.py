"""Unit tests for compute.models.initialization.nuclear_repulsion_energy module.

Tests cover:
- Basic energy computation for H2 and water molecules
- Scaling with nuclear charge Z (via pairwise_energy)
- Pairwise energy calculation
- Edge case: coincident nuclei
- __repr__ and __float__ methods

All tests use pytest with parametrize, no test classes.
"""

import math
from typing import List, Tuple

import pytest

from compute.models.initialization import NuclearRepulsionEnergy
from compute.models.molecule import Molecule
from tests.unit_tests.utilities import h2_molecule, water_molecule

# ======================================================================
# Helper Functions
# ======================================================================


def _manual_e_nuc(nuclei: List[Tuple[int, Tuple[float, float, float]]]) -> float:
    """Manually compute E_nuc from list of (Z, (x, y, z)) tuples.

    :param nuclei: List of nuclear charges and positions in Bohr.
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :returns: Nuclear repulsion energy in Hartree.
    :rtype: float
    """
    e_nuc = 0.0
    n = len(nuclei)
    for a in range(n):
        Z_a, R_a = nuclei[a]
        for b in range(a + 1, n):
            Z_b, R_b = nuclei[b]
            R_ab = math.sqrt(
                (R_a[0] - R_b[0]) ** 2
                + (R_a[1] - R_b[1]) ** 2
                + (R_a[2] - R_b[2]) ** 2
            )
            e_nuc += Z_a * Z_b / R_ab
    return e_nuc


# ======================================================================
# Basic Functionality Tests
# ======================================================================


def test_h2_nuclear_repulsion(h2_molecule: Molecule) -> None:
    """Verify E_nuc for H2 at ~0.74 Angstrom bond length.

    :param h2_molecule: H2 molecule fixture from tests.unit_tests.utils.
    :type h2_molecule: Molecule
    """
    nuc_rep = NuclearRepulsionEnergy(h2_molecule)

    assert nuc_rep.n_atoms == 2
    assert nuc_rep.energy > 0  # Always repulsive

    # Manual check: E = Z_A * Z_B / R_AB = 1 * 1 / R
    R_ab = abs(nuc_rep.nuclei[0][1][2] - nuc_rep.nuclei[1][1][2])
    expected = 1.0 / R_ab
    assert nuc_rep.energy == pytest.approx(expected, rel=1e-10)


def test_water_nuclear_repulsion(water_molecule: Molecule) -> None:
    """Verify E_nuc for water molecule (3 atoms).

    :param water_molecule: Water molecule fixture from tests.unit_tests.utils.
    :type water_molecule: Molecule
    """
    nuc_rep = NuclearRepulsionEnergy(water_molecule)

    assert nuc_rep.n_atoms == 3
    assert nuc_rep.energy > 0

    # Verify against manual calculation
    expected = _manual_e_nuc(nuc_rep.nuclei)
    assert nuc_rep.energy == pytest.approx(expected, rel=1e-10)


# ======================================================================
# Scaling Tests
# ======================================================================


@pytest.mark.parametrize(
    "Z_pair, expected_ratio",
    [
        ((1, 1), 1.0),  # H-H
        ((2, 1), 2.0),  # He-H (Z=2 vs Z=1)
        ((3, 1), 3.0),  # Li-H (Z=3 vs Z=1)
        ((6, 1), 6.0),  # C-H (Z=6 vs Z=1)
        ((8, 1), 8.0),  # O-H (Z=8 vs Z=1)
    ],
    ids=["H-H", "He-H", "Li-H", "C-H", "O-H"],
)
def test_scaling_with_nuclear_charge(
    Z_pair: Tuple[int, int], expected_ratio: float
) -> None:
    """Verify E_nuc scales linearly with Z_A * Z_B.

    :param Z_pair: Tuple of (Z_A, Z_B) for comparison.
    :param expected_ratio: Expected ratio E(Z_A, Z_B) / E(H-H).
    """
    # Reference: H-H at fixed distance
    R_a: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_b: Tuple[float, float, float] = (0.0, 0.0, 2.0)  # Fixed distance in Bohr

    E_hh = NuclearRepulsionEnergy.pairwise_energy(1, R_a, 1, R_b)
    E_test = NuclearRepulsionEnergy.pairwise_energy(Z_pair[0], R_a, Z_pair[1], R_b)

    actual_ratio = E_test / E_hh
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)


# ======================================================================
# Pairwise Energy Tests
# ======================================================================


def test_pairwise_energy_basic() -> None:
    """Verify pairwise_energy static method."""
    R_a: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_b: Tuple[float, float, float] = (0.0, 0.0, 2.0)  # 2 Bohr apart

    E = NuclearRepulsionEnergy.pairwise_energy(1, R_a, 1, R_b)

    assert E == pytest.approx(0.5, rel=1e-10)  # 1 * 1 / 2 = 0.5


def test_pairwise_energy_diagonal() -> None:
    """Verify pairwise_energy in diagonal direction."""
    R_a: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_b: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    E = NuclearRepulsionEnergy.pairwise_energy(1, R_a, 1, R_b)

    R = math.sqrt(3.0)  # sqrt(1^2 + 1^2 + 1^2)
    expected = 1.0 / R
    assert E == pytest.approx(expected, rel=1e-10)


def test_pairwise_energy_symmetric() -> None:
    """Verify pairwise_energy is symmetric: E(A,B) = E(B,A)."""
    R_a: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_b: Tuple[float, float, float] = (1.5, 2.3, 0.7)

    E_ab = NuclearRepulsionEnergy.pairwise_energy(6, R_a, 8, R_b)
    E_ba = NuclearRepulsionEnergy.pairwise_energy(8, R_b, 6, R_a)

    assert E_ab == pytest.approx(E_ba, rel=1e-10)


# ======================================================================
# Edge Cases
# ======================================================================


def test_coincident_nuclei_raises() -> None:
    """Verify ValueError when nuclei are at the same position."""
    R_a: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    R_b: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="same position"):
        NuclearRepulsionEnergy.pairwise_energy(1, R_a, 1, R_b)


# ======================================================================
# Method Tests (__repr__, __float__)
# ======================================================================


def test_repr(h2_molecule: Molecule) -> None:
    """Verify __repr__ contains expected information.

    :param h2_molecule: H2 molecule fixture from tests.unit_tests.utils.
    :type h2_molecule: Molecule
    """
    nuc_rep = NuclearRepulsionEnergy(h2_molecule)

    repr_str = repr(nuc_rep)
    assert "NuclearRepulsionEnergy" in repr_str
    assert "n_atoms=2" in repr_str
    assert "energy=" in repr_str


def test_float_conversion(h2_molecule: Molecule) -> None:
    """Verify __float__ returns the energy value.

    :param h2_molecule: H2 molecule fixture from tests.unit_tests.utils.
    :type h2_molecule: Molecule
    """
    nuc_rep = NuclearRepulsionEnergy(h2_molecule)

    assert float(nuc_rep) == nuc_rep.energy
    assert isinstance(float(nuc_rep), float)


# ======================================================================
# Nuclei Data Test
# ======================================================================


def test_nuclei_data_correct(water_molecule: Molecule) -> None:
    """Verify nuclei list contains correct charges and positions.

    :param water_molecule: Water molecule fixture from tests.unit_tests.utils.
    :type water_molecule: Molecule
    """
    nuc_rep = NuclearRepulsionEnergy(water_molecule)

    # Water has O (Z=8) and two H (Z=1)
    charges = [n[0] for n in nuc_rep.nuclei]
    assert 8 in charges  # Oxygen
    assert charges.count(1) == 2  # Two hydrogens
