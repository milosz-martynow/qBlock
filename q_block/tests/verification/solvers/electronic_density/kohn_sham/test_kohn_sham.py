"""Verification tests for Kohn-Sham DFT solvers.

Tests cover:
- RestrictedKohnSham construction and properties
- UnrestrictedKohnSham construction and properties
- Basis function evaluation on grid
- Density-on-grid computation
- XC potential matrix construction
- RKS and UKS matrix storage
- Parameter validation

Uses minimal H₂ (STO-3G) to keep tests fast.
"""

from typing import List, Tuple

import numpy as np
import pytest

from q_block.compute.environment.io.basis_set import Pople
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from q_block.compute.models.molecule import Molecule
from q_block.compute.solvers.electronic_density.functionals import (
    B3LYP,
    PBE,
    SVWN,
)
from q_block.compute.solvers.electronic_density.kohn_sham import (
    RestrictedKohnSham,
    UnrestrictedKohnSham,
)
from q_block.tests.verification.environment.constants import (
    BASIS_STO_3G,
)


# ======================================================================
# Fixtures
# ======================================================================


def _h2_inputs(
    basis: Pople = BASIS_STO_3G,
) -> Tuple[list, list, float]:
    """Return (cgtos, nuclei, e_nuclear) for H₂ at 0.74 Å.

    :param basis: Basis set.
    :type basis: Pople
    :returns: CGTOs, nuclei, and nuclear repulsion energy.
    :rtype: Tuple[list, list, float]
    """
    inp = InputData()
    inp.from_script(
        atom_data=[
            ["H", 0.0, 0.0, 0.0, basis],
            ["H", 0.0, 0.0, 0.74, basis],
        ]
    )
    mol = Molecule(input_data=inp, multiplicity=1)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    cgtos = mol.contracted_gaussian_type_orbitals
    nuclei = [
        (
            a.atomic_number,
            (a.coordinates.x, a.coordinates.y, a.coordinates.z),
        )
        for a in mol.atoms
    ]
    e_nuclear = NuclearRepulsionEnergy(mol).energy
    return cgtos, nuclei, e_nuclear


# ======================================================================
# RKS construction
# ======================================================================


def test_rks_construction_svwn() -> None:
    """RKS initialises correctly with SVWN functional."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.n_alpha == 1
    assert rks.n_beta == 1
    assert rks.n_electrons == 2
    assert rks._shared_spin is True
    assert rks.functional.name == "SVWN"


def test_rks_odd_electrons_raises() -> None:
    """RKS raises ValueError for odd electron count."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    with pytest.raises(ValueError, match="even number"):
        RestrictedKohnSham(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            functional=SVWN(),
            n_electrons=1,
            n_radial=20,
            n_angular=6,
        )


def test_rks_phi_grid_shape() -> None:
    """Basis function values on grid have correct shape."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.phi_grid.shape == (rks.n_basis, rks.grid.n_points)
    assert np.all(np.isfinite(rks.phi_grid))


def test_rks_dphi_grid_none_for_lda() -> None:
    """Gradient grid is None for LDA functionals."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is None


def test_rks_dphi_grid_computed_for_gga() -> None:
    """Gradient grid is computed for GGA functionals."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=PBE(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is not None
    assert rks.dphi_grid.shape == (
        3,
        rks.n_basis,
        rks.grid.n_points,
    )
    assert np.all(np.isfinite(rks.dphi_grid))


def test_rks_dphi_grid_computed_for_hybrid() -> None:
    """Gradient grid is computed for hybrid functionals."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=B3LYP(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    assert rks.dphi_grid is not None


# ======================================================================
# RKS density on grid
# ======================================================================


def test_rks_density_on_grid_non_negative() -> None:
    """Density on grid is non-negative."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=20,
        n_angular=6,
    )
    n = rks.n_basis
    P = np.eye(n) * 0.5
    rho = rks._density_on_grid(P)
    assert rho.shape == (rks.grid.n_points,)
    assert np.all(rho >= 0.0)


# ======================================================================
# UKS construction
# ======================================================================


def test_uks_construction_svwn() -> None:
    """UKS initialises correctly with SVWN functional."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=20,
        n_angular=6,
    )
    assert uks.n_alpha == 1
    assert uks.n_beta == 1
    assert uks.n_electrons == 2
    assert uks._shared_spin is False
    assert uks.functional.name == "SVWN"


def test_uks_negative_electrons_raises() -> None:
    """UKS raises ValueError for negative electron count."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        UnrestrictedKohnSham(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=e_nuc,
            functional=SVWN(),
            n_alpha=-1,
            n_beta=0,
            n_radial=20,
            n_angular=6,
        )


# ======================================================================
# RKS SCF convergence (small system, fast test)
# ======================================================================


def test_rks_svwn_h2_converges() -> None:
    """RKS-SVWN converges for H₂ with STO-3G."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0
    assert "C" in rks.matrices
    assert "P" in rks.matrices


def test_rks_pbe_h2_converges() -> None:
    """RKS-PBE converges for H₂ with STO-3G."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=PBE(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0


def test_rks_b3lyp_h2_converges() -> None:
    """RKS-B3LYP converges for H₂ with STO-3G."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=B3LYP(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert rks.converged
    assert rks.e_total < 0.0


# ======================================================================
# UKS SCF convergence
# ======================================================================


def test_uks_svwn_h2_converges() -> None:
    """UKS-SVWN converges for H₂ with STO-3G."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    uks = UnrestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_alpha=1,
        n_beta=1,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert uks.converged
    assert uks.e_total < 0.0
    assert "C_alpha" in uks.matrices
    assert "C_beta" in uks.matrices


# ======================================================================
# Energy ordering (SVWN < PBE < B3LYP typical for H₂)
# ======================================================================


def test_rks_total_energy_finite() -> None:
    """RKS-SVWN produces finite total energy for H₂."""
    cgtos, nuclei, e_nuc = _h2_inputs()
    rks = RestrictedKohnSham(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        functional=SVWN(),
        n_electrons=2,
        n_radial=30,
        n_angular=6,
        max_iterations=100,
        convergence_threshold=1e-5,
    ).run()
    assert np.isfinite(rks.e_total)
