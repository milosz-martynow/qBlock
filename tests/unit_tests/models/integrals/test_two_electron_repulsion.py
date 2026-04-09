"""Unit tests for q_block.models.integrals.two_electron_repulsion module.

Tests cover:
- Primitive two-electron repulsion integrals (ERIs)
- Contracted two-electron repulsion integrals
- Full ERI tensor for molecules
- 8-fold permutational symmetry
- Positive values for all ERIs (Coulomb repulsion is positive)
- Coulomb (J) and Exchange (K) integral properties

All tests use pytest with parametrize, no test classes.
"""

import numpy as np
import pytest

from q_block import ContractedGaussianTypeOrbital, Molecule, TwoElectronRepulsion, Overlap
from tests.unit_tests.constants import ORIGIN
from tests.unit_tests.utils import (
    h2_molecule,
    water_molecule,
)


# ======================================================================
# Primitive ERI Tests
# ======================================================================


def test_primitive_eri_self_positive() -> None:
    """Verify ERI of four identical s-orbitals is positive.

    (ss|ss) should be a positive Coulomb repulsion integral.
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0

    eri: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=A, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=A, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=A, lx4=0, ly4=0, lz4=0,
    )
    assert eri > 0


def test_primitive_eri_two_center_positive() -> None:
    """Verify ERI with two different centers is positive."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (1.0, 0.0, 0.0)
    alpha: float = 1.0

    eri: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=A, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=B, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=B, lx4=0, ly4=0, lz4=0,
    )
    assert eri > 0


def test_primitive_eri_four_center_positive() -> None:
    """Verify ERI with four different centers is positive (or zero)."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (1.0, 0.0, 0.0)
    C: tuple[float, float, float] = (0.0, 1.0, 0.0)
    D: tuple[float, float, float] = (1.0, 1.0, 0.0)
    alpha: float = 1.0

    eri: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=B, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=C, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=D, lx4=0, ly4=0, lz4=0,
    )
    # ERIs are always non-negative (unless numerical precision issues)
    assert eri >= -1e-12


def test_primitive_eri_far_apart_negligible() -> None:
    """Verify ERIs with far apart orbitals are negligible."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (0.0, 0.0, 0.0)
    C: tuple[float, float, float] = (100.0, 0.0, 0.0)
    D: tuple[float, float, float] = (100.0, 0.0, 0.0)
    alpha: float = 1.0

    eri: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=B, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=C, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=D, lx4=0, ly4=0, lz4=0,
    )
    # Far apart orbitals should have near-zero ERI
    assert abs(eri) < 0.1


def test_primitive_eri_decreases_with_distance() -> None:
    """Verify ERI decreases as centers are moved apart."""
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0

    # Both at origin
    eri_close: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=A, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=A, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=A, lx4=0, ly4=0, lz4=0,
    )

    # Separated by 2 Bohr
    B_far: tuple[float, float, float] = (2.0, 0.0, 0.0)
    eri_far: float = TwoElectronRepulsion.primitive_eri(
        alpha=alpha, A=A, lx1=0, ly1=0, lz1=0,
        beta=alpha, B=A, lx2=0, ly2=0, lz2=0,
        gamma=alpha, C=B_far, lx3=0, ly3=0, lz3=0,
        delta=alpha, D=B_far, lx4=0, ly4=0, lz4=0,
    )

    assert eri_close > eri_far


# ----------------------------------------------------------------------
# 8-fold Symmetry Tests for Primitive ERI
# ----------------------------------------------------------------------


def test_primitive_eri_8fold_symmetry() -> None:
    """Verify (μν|λσ) has 8-fold permutational symmetry.

    (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)
    """
    A: tuple[float, float, float] = (0.0, 0.0, 0.0)
    B: tuple[float, float, float] = (1.0, 0.0, 0.0)
    C: tuple[float, float, float] = (0.0, 1.0, 0.0)
    D: tuple[float, float, float] = (0.5, 0.5, 0.0)

    alpha, beta, gamma, delta = 1.5, 1.2, 1.8, 1.0
    lx1, ly1, lz1 = 0, 0, 0
    lx2, ly2, lz2 = 1, 0, 0
    lx3, ly3, lz3 = 0, 1, 0
    lx4, ly4, lz4 = 0, 0, 1

    # Original: (μν|λσ)
    eri_original = TwoElectronRepulsion.primitive_eri(
        alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2,
        gamma, C, lx3, ly3, lz3, delta, D, lx4, ly4, lz4,
    )

    # (νμ|λσ) - swap first two
    eri_swap_12 = TwoElectronRepulsion.primitive_eri(
        beta, B, lx2, ly2, lz2, alpha, A, lx1, ly1, lz1,
        gamma, C, lx3, ly3, lz3, delta, D, lx4, ly4, lz4,
    )

    # (μν|σλ) - swap last two
    eri_swap_34 = TwoElectronRepulsion.primitive_eri(
        alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2,
        delta, D, lx4, ly4, lz4, gamma, C, lx3, ly3, lz3,
    )

    # (λσ|μν) - swap bra and ket
    eri_swap_braket = TwoElectronRepulsion.primitive_eri(
        gamma, C, lx3, ly3, lz3, delta, D, lx4, ly4, lz4,
        alpha, A, lx1, ly1, lz1, beta, B, lx2, ly2, lz2,
    )

    assert eri_swap_12 == pytest.approx(expected=eri_original, rel=1e-10)
    assert eri_swap_34 == pytest.approx(expected=eri_original, rel=1e-10)
    assert eri_swap_braket == pytest.approx(expected=eri_original, rel=1e-10)


# ======================================================================
# Contracted ERI Tests
# ======================================================================


def test_contracted_eri_single_primitive_equals_primitive() -> None:
    """Verify single-primitive CGTO ERI matches primitive ERI."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )

    eri_contracted: float = TwoElectronRepulsion.contracted_eri(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0,
        cgto3=cgto, lx3=0, ly3=0, lz3=0,
        cgto4=cgto, lx4=0, ly4=0, lz4=0,
    )
    A = (0.0, 0.0, 0.0)
    eri_primitive: float = TwoElectronRepulsion.primitive_eri(
        alpha=1.0, A=A, lx1=0, ly1=0, lz1=0,
        beta=1.0, B=A, lx2=0, ly2=0, lz2=0,
        gamma=1.0, C=A, lx3=0, ly3=0, lz3=0,
        delta=1.0, D=A, lx4=0, ly4=0, lz4=0,
    )
    assert eri_contracted == pytest.approx(expected=eri_primitive, rel=1e-10)


def test_contracted_eri_positive() -> None:
    """Verify contracted ERI self-integral is positive."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[3.0, 1.0, 0.3],
        contractions=[0.5, 0.3, 0.2],
    )

    eri: float = TwoElectronRepulsion.contracted_eri(
        cgto1=cgto, lx1=0, ly1=0, lz1=0,
        cgto2=cgto, lx2=0, ly2=0, lz2=0,
        cgto3=cgto, lx3=0, ly3=0, lz3=0,
        cgto4=cgto, lx4=0, ly4=0, lz4=0,
    )
    assert eri > 0


# ======================================================================
# Tensor Initialization Tests
# ======================================================================


def test_eri_tensor_empty_basis_raises() -> None:
    """Verify empty basis set raises ValueError."""
    with pytest.raises(expected_exception=ValueError, match="empty basis set"):
        TwoElectronRepulsion(cgtos=[])


def test_eri_tensor_single_s_orbital() -> None:
    """Verify single s-orbital gives (1,1,1,1) tensor."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=0,
        exponents=[1.0],
        contractions=[1.0],
    )
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(cgtos=[cgto])
    assert ERI.n_basis == 1
    assert ERI.tensor.shape == (1, 1, 1, 1)


def test_eri_tensor_single_p_orbital() -> None:
    """Verify single p-shell gives (3,3,3,3) tensor."""
    cgto: ContractedGaussianTypeOrbital = ContractedGaussianTypeOrbital(
        center=ORIGIN,
        l=1,
        exponents=[1.0],
        contractions=[1.0],
    )
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(cgtos=[cgto])
    assert ERI.n_basis == 3
    assert ERI.tensor.shape == (3, 3, 3, 3)


# ======================================================================
# 8-fold Symmetry Tests for Full Tensor
# ======================================================================


def test_eri_tensor_8fold_symmetry_h2(h2_molecule: Molecule) -> None:
    """Verify ERI tensor has 8-fold permutational symmetry for H2.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )

    n = ERI.n_basis
    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    eri_val = ERI.tensor[mu, nu, lam, sig]

                    # Check all 8 permutations
                    assert ERI.tensor[nu, mu, lam, sig] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[mu, nu, sig, lam] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[nu, mu, sig, lam] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[lam, sig, mu, nu] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[sig, lam, mu, nu] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[lam, sig, nu, mu] == pytest.approx(eri_val, rel=1e-10)
                    assert ERI.tensor[sig, lam, nu, mu] == pytest.approx(eri_val, rel=1e-10)


# ======================================================================
# Coulomb and Exchange Tests
# ======================================================================


def test_eri_coulomb_positive_h2(h2_molecule: Molecule) -> None:
    """Verify Coulomb integrals (μμ|νν) are positive.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )

    for mu in range(ERI.n_basis):
        for nu in range(ERI.n_basis):
            # Coulomb integral J[μ,ν] = (μμ|νν)
            J_mn = ERI.tensor[mu, mu, nu, nu]
            assert J_mn > 0, f"Coulomb J[{mu},{nu}] = {J_mn} should be positive"


def test_eri_exchange_positive_h2(h2_molecule: Molecule) -> None:
    """Verify Exchange integrals (μν|μν) are positive or zero.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )

    for mu in range(ERI.n_basis):
        for nu in range(ERI.n_basis):
            # Exchange integral K[μ,ν] = (μν|μν)
            K_mn = ERI.tensor[mu, nu, mu, nu]
            assert K_mn >= -1e-12, f"Exchange K[{mu},{nu}] = {K_mn} should be >= 0"


def test_eri_coulomb_greater_than_exchange(h2_molecule: Molecule) -> None:
    """Verify Coulomb >= Exchange for same orbital pairs.

    For μ ≠ ν: (μμ|νν) >= (μν|μν) due to the nature of the two-electron integral.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )

    for mu in range(ERI.n_basis):
        for nu in range(mu + 1, ERI.n_basis):  # Only off-diagonal
            J = ERI.tensor[mu, mu, nu, nu]  # Coulomb
            K = ERI.tensor[mu, nu, mu, nu]  # Exchange
            assert J >= K - 1e-10, f"Coulomb J[{mu},{nu}]={J} should be >= Exchange K[{mu},{nu}]={K}"


# ======================================================================
# Matrix/Tensor Properties Tests
# ======================================================================


def test_eri_tensor_repr(h2_molecule: Molecule) -> None:
    """Verify __repr__ output.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )
    repr_str: str = repr(ERI)
    assert "TwoElectronRepulsion" in repr_str
    assert "n_basis" in repr_str


def test_eri_tensor_getitem(h2_molecule: Molecule) -> None:
    """Verify __getitem__ access.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )
    assert ERI[0, 0, 0, 0] == ERI.tensor[0, 0, 0, 0]
    assert ERI[0, 1, 0, 1] == ERI.tensor[0, 1, 0, 1]


# ======================================================================
# H2 Specific Tests
# ======================================================================


def test_h2_eri_tensor_dimension(h2_molecule: Molecule) -> None:
    """Verify H2 with STO-3G has 2 basis functions.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )
    assert ERI.n_basis == 2
    assert ERI.tensor.shape == (2, 2, 2, 2)


def test_h2_eri_tensor_values_physical(h2_molecule: Molecule) -> None:
    """Verify H2 ERI values are in physically reasonable range.

    For H2 with STO-3G, the ERIs should be in the range [0, 1] Hartree.

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )

    for mu in range(ERI.n_basis):
        for nu in range(ERI.n_basis):
            for lam in range(ERI.n_basis):
                for sig in range(ERI.n_basis):
                    eri_val = ERI.tensor[mu, nu, lam, sig]
                    assert eri_val >= -1e-12, f"ERI should be non-negative"
                    assert eri_val <= 2.0, f"ERI should be < 2 Hartree for H2"


# ======================================================================
# Water Specific Tests
# ======================================================================


def test_water_eri_tensor_dimension(water_molecule: Molecule) -> None:
    """Verify water with STO-3G has 7 basis functions.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=water_molecule.contracted_gaussian_type_orbitals
    )
    # O: 1s, 2s, 2px, 2py, 2pz = 5 functions
    # H1: 1s = 1 function
    # H2: 1s = 1 function
    # Total: 7
    assert ERI.n_basis == 7
    assert ERI.tensor.shape == (7, 7, 7, 7)


def test_water_eri_tensor_8fold_symmetry_sample(water_molecule: Molecule) -> None:
    """Verify sample ERIs satisfy 8-fold symmetry for water.

    Full verification would be too slow, so test sample indices.

    :param water_molecule: Molecule - Water molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=water_molecule.contracted_gaussian_type_orbitals
    )

    # Test sample indices
    test_indices = [(0, 1, 2, 3), (1, 2, 3, 4), (0, 3, 1, 5), (2, 4, 6, 0)]

    for mu, nu, lam, sig in test_indices:
        eri_val = ERI.tensor[mu, nu, lam, sig]

        # Check all 8 permutations
        assert ERI.tensor[nu, mu, lam, sig] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[mu, nu, sig, lam] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[nu, mu, sig, lam] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[lam, sig, mu, nu] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[sig, lam, mu, nu] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[lam, sig, nu, mu] == pytest.approx(eri_val, rel=1e-10)
        assert ERI.tensor[sig, lam, nu, mu] == pytest.approx(eri_val, rel=1e-10)


# ======================================================================
# Comparison with Other Integrals
# ======================================================================


def test_eri_vs_overlap_different_dimensions(h2_molecule: Molecule) -> None:
    """Verify ERI tensor (4D) is different from Overlap matrix (2D).

    :param h2_molecule: Molecule - H2 molecule fixture with STO-3G basis.
    """
    ERI: TwoElectronRepulsion = TwoElectronRepulsion(
        cgtos=h2_molecule.contracted_gaussian_type_orbitals
    )
    S: Overlap = Overlap(cgtos=h2_molecule.contracted_gaussian_type_orbitals)

    assert len(ERI.tensor.shape) == 4  # 4D tensor
    assert len(S.matrix.shape) == 2  # 2D matrix
    assert ERI.n_basis == S.n_basis
