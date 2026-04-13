"""Unit tests for compute.solvers.scf module.

Tests cover:
- SCF construction and parameter validation
- Integral matrix shapes and symmetry
- Orthogonalisation matrix X properties
- DIIS and error metric initialisation
- Result attributes before run()
- Concrete subclass run (convergence) via a minimal RHF stub

All tests use pytest with parametrize, no test classes.
"""

from typing import List, Tuple

import numpy as np
import pytest

from compute import Molecule
from compute.models.basis_functions import ContractedGaussianTypeOrbital
from compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from compute.solvers.calculation_error_metric import CalculationErrorMetric
from compute.solvers.diis import DIIS
from compute.solvers.scf import SCF
from tests.verification.utilities import extract_nuclei, h2_molecule

# ======================================================================
# Minimal concrete SCF subclass (RHF-like stub)
# ======================================================================


class _StubRHF(SCF):
    """Minimal concrete RHF implementation for testing the SCF template.

    Uses the simplest closed-shell formulas so the SCF loop can be
    exercised without requiring a full RHF module.  This stub implements
    all five abstract methods of the SCF base class with textbook
    closed-shell Hartree-Fock expressions: four-index Coulomb and
    exchange integrals for the Fock matrix, a half-trace for the
    electronic energy, and an occupation-based density matrix.  Results
    are collected into public attributes so that tests can inspect every
    intermediate quantity produced by the SCF cycle.
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        n_electrons: int,
        **kwargs,
    ) -> None:
        """Initialise the stub with basis functions, nuclei, and electron count.

        :param cgtos: Contracted Gaussian basis functions for the molecule.
        :type cgtos: List[ContractedGaussianTypeOrbital]
        :param nuclei: Nuclear charges and Cartesian coordinates as (Z, (x, y, z)) pairs.
        :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
        :param e_nuclear: Nuclear repulsion energy in Hartree.
        :type e_nuclear: float
        :param n_electrons: Total number of electrons (must be even for closed-shell).
        :type n_electrons: int
        :param kwargs: Extra keyword arguments forwarded to the SCF base class.
        :type kwargs: dict
        """
        super().__init__(cgtos, nuclei, e_nuclear, **kwargs)
        # Number of doubly occupied orbitals for closed-shell RHF.
        self.n_occ: int = n_electrons // 2

    def _build_fock(self, density: np.ndarray) -> np.ndarray:
        """Build the Fock matrix from the density matrix.

        Constructs the Coulomb (J) and exchange (K) matrices via a
        four-index contraction over the electron-repulsion integrals
        and returns H_core + J - 0.5 * K.

        :param density: Current density matrix of shape (n_basis, n_basis).
        :type density: np.ndarray
        :returns: Fock matrix of shape (n_basis, n_basis).
        :rtype: np.ndarray
        """
        n = self.n_basis
        # J is the Coulomb matrix, K is the exchange matrix.
        J = np.zeros((n, n))
        K = np.zeros((n, n))
        for mu in range(n):
            for nu in range(n):
                for lam in range(n):
                    for sig in range(n):
                        J[mu, nu] += density[lam, sig] * self.eri[mu, nu, lam, sig]
                        K[mu, nu] += density[lam, sig] * self.eri[mu, lam, nu, sig]
        return self.H + J - 0.5 * K

    def _compute_electronic_energy(
        self, density: np.ndarray, fock: np.ndarray
    ) -> float:
        """Compute the one-electron plus two-electron electronic energy.

        Uses the standard RHF trace formula E_el = 0.5 * Tr[P (H + F)].

        :param density: Current density matrix of shape (n_basis, n_basis).
        :type density: np.ndarray
        :param fock: Current Fock matrix of shape (n_basis, n_basis).
        :type fock: np.ndarray
        :returns: Electronic energy as a scalar float in Hartree.
        :rtype: float
        """
        return 0.5 * np.sum(density * (self.H + fock))

    def _build_density(self, C: np.ndarray) -> np.ndarray:
        """Build the closed-shell density matrix P = 2 C_occ C_occ^T.

        Selects the first n_occ columns of the coefficient matrix
        corresponding to the occupied molecular orbitals.

        :param C: MO coefficient matrix of shape (n_basis, n_basis).
        :type C: np.ndarray
        :returns: Density matrix of shape (n_basis, n_basis).
        :rtype: np.ndarray
        """
        # Slice to keep only the occupied MO coefficients.
        C_occ = C[:, : self.n_occ]
        return 2.0 * C_occ @ C_occ.T

    def _initial_density(self, C: np.ndarray) -> np.ndarray:
        """Return the initial density guess, identical to _build_density.

        :param C: MO coefficient matrix from the initial diagonalisation.
        :type C: np.ndarray
        :returns: Initial density matrix of shape (n_basis, n_basis).
        :rtype: np.ndarray
        """
        return self._build_density(C)

    def _collect_results(
        self,
        converged: bool,
        n_iterations: int,
        e_electronic: float,
        fock,
        density,
        C,
        epsilon,
    ) -> None:
        """Store all SCF results as public attributes for test inspection.

        Saves convergence status, iteration count, energies, and all
        key matrices (overlap, core Hamiltonian, Fock, density,
        coefficients, orbital energies) in a single dictionary.

        :param converged: Whether the SCF loop met the convergence criterion.
        :type converged: bool
        :param n_iterations: Number of SCF iterations performed.
        :type n_iterations: int
        :param e_electronic: Final electronic energy in Hartree.
        :type e_electronic: float
        :param fock: Converged Fock matrix.
        :type fock: np.ndarray
        :param density: Converged density matrix.
        :type density: np.ndarray
        :param C: Converged MO coefficient matrix.
        :type C: np.ndarray
        :param epsilon: Orbital energies (eigenvalues) array.
        :type epsilon: np.ndarray
        """
        self.converged = converged
        self.n_iterations = n_iterations
        self.e_electronic = e_electronic
        # Total energy = electronic + nuclear repulsion.
        self.e_total = e_electronic + self.e_nuclear
        self.matrices = {
            "S": self.S,
            "H": self.H,
            "F": fock,
            "P": density,
            "C": C,
            "epsilon": epsilon,
        }


# ======================================================================
# Fixtures
# ======================================================================


# h2_molecule is a pytest fixture imported from tests.verification.utilities (defined with
# @pytest.fixture there).  Importing it into this module is enough for
# pytest to discover it and inject it into any test that declares an
# ``h2_molecule`` parameter.


@pytest.fixture
def h2_scf(h2_molecule: Molecule) -> _StubRHF:
    """Uninitialised _StubRHF for H2 / STO-3G (not yet run).

    Extracts nuclei positions, basis functions, and the nuclear
    repulsion energy from the shared h2_molecule fixture.  The
    returned object is ready for .run() but has not been executed.

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    :returns: Uninitialised _StubRHF instance ready for .run().
    :rtype: _StubRHF
    """

    # Extract nuclear coordinates as (Z, (x, y, z)) pairs.
    nuclei = extract_nuclei(h2_molecule)
    # Contracted Gaussian basis set for the molecule.
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    # Nuclear repulsion energy computed once and reused.
    e_nuc = NuclearRepulsionEnergy(h2_molecule).energy
    return _StubRHF(
        cgtos=cgtos,
        nuclei=nuclei,
        e_nuclear=e_nuc,
        n_electrons=2,
    )


# ======================================================================
# Construction & Validation Tests
# ======================================================================


def test_max_iterations_validation(h2_molecule: Molecule) -> None:
    """max_iterations < 1 should raise ValueError.

    Passing max_iterations=0 is invalid because the SCF loop must
    execute at least one iteration.  The constructor should reject
    this value early to prevent silent misconfiguration.

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    """
    # Prepare nuclei and basis from the shared H2 fixture.
    nuclei = extract_nuclei(h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        _StubRHF(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=0.0,
            n_electrons=2,
            max_iterations=0,
        )


def test_convergence_threshold_validation(h2_molecule: Molecule) -> None:
    """convergence_threshold <= 0 should raise ValueError.

    A non-positive threshold would make the convergence check
    meaningless or always satisfied.  The constructor must reject
    such values to guarantee a well-defined stopping criterion.

    :param h2_molecule: Pytest fixture providing an H2 Molecule with STO-3G basis in Bohr.
    :type h2_molecule: Molecule
    """
    # Prepare nuclei and basis from the shared H2 fixture.
    nuclei = extract_nuclei(h2_molecule)
    cgtos = h2_molecule.contracted_gaussian_type_orbitals
    with pytest.raises(ValueError, match="convergence_threshold must be > 0"):
        _StubRHF(
            cgtos=cgtos,
            nuclei=nuclei,
            e_nuclear=0.0,
            n_electrons=2,
            convergence_threshold=-1e-8,
        )


# ======================================================================
# Integral Matrix Tests
# ======================================================================


def test_overlap_shape(h2_scf: _StubRHF) -> None:
    """Overlap matrix should be n_basis × n_basis.

    The overlap matrix S measures the non-orthogonality of the
    Gaussian basis functions.  Its shape must match the total
    number of basis functions in the molecule.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # n is the total number of contracted Gaussian basis functions.
    n = h2_scf.n_basis
    assert h2_scf.S.shape == (n, n)


def test_overlap_symmetry(h2_scf: _StubRHF) -> None:
    """Overlap matrix should be symmetric.

    By definition S_mu_nu = S_nu_mu because the integral of
    basis function products is independent of ordering.  Any
    asymmetry indicates a bug in integral evaluation.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    np.testing.assert_allclose(h2_scf.S, h2_scf.S.T, atol=1e-14)


def test_overlap_positive_definite(h2_scf: _StubRHF) -> None:
    """Overlap matrix should be positive definite.

    Positive definiteness is required for the canonical
    orthogonalisation procedure (S^{-1/2}).  If any eigenvalue
    is zero or negative, the basis is linearly dependent.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # All eigenvalues of S must be strictly positive.
    eigvals = np.linalg.eigvalsh(h2_scf.S)
    assert np.all(eigvals > 0)


def test_kinetic_shape(h2_scf: _StubRHF) -> None:
    """Kinetic energy matrix should be n_basis × n_basis.

    The kinetic energy integrals form a square matrix over the
    basis set.  Shape mismatch would indicate an error in the
    integral engine loop bounds.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # n is the dimension of the one-electron integral matrices.
    n = h2_scf.n_basis
    assert h2_scf.T.shape == (n, n)


def test_kinetic_symmetry(h2_scf: _StubRHF) -> None:
    """Kinetic energy matrix should be symmetric.

    The kinetic energy operator is Hermitian, so T_mu_nu must
    equal T_nu_mu for all basis function pairs.  Asymmetry
    signals incorrect derivative handling in the integrals.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    np.testing.assert_allclose(h2_scf.T, h2_scf.T.T, atol=1e-14)


def test_nuclear_attraction_shape(h2_scf: _StubRHF) -> None:
    """Nuclear attraction matrix should be n_basis × n_basis.

    The nuclear attraction matrix V contains the one-electron
    Coulomb integrals between each basis pair and every nucleus.
    Its dimensions must equal the number of basis functions.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # n is the total number of basis functions.
    n = h2_scf.n_basis
    assert h2_scf.V.shape == (n, n)


def test_nuclear_attraction_symmetry(h2_scf: _StubRHF) -> None:
    """Nuclear attraction matrix should be symmetric.

    The nuclear attraction operator is Hermitian, guaranteeing
    V_mu_nu = V_nu_mu.  Breaking this symmetry would produce
    incorrect core Hamiltonian and Fock matrices.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    np.testing.assert_allclose(h2_scf.V, h2_scf.V.T, atol=1e-14)


def test_core_hamiltonian_equals_T_plus_V(h2_scf: _StubRHF) -> None:
    """Core Hamiltonian H should equal T + V.

    The one-electron core Hamiltonian is defined as the sum of
    the kinetic energy and nuclear attraction integrals.  Any
    discrepancy indicates an error in the SCF initialisation.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    np.testing.assert_allclose(h2_scf.H, h2_scf.T + h2_scf.V, atol=1e-14)


def test_eri_shape(h2_scf: _StubRHF) -> None:
    """ERI tensor should be (n, n, n, n).

    The electron-repulsion integral tensor has four indices,
    one per basis function in the two-electron interaction.
    Incorrect shape would cause index errors in _build_fock.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # n is the number of basis functions; ERI is a rank-4 tensor.
    n = h2_scf.n_basis
    assert h2_scf.eri.shape == (n, n, n, n)


# ======================================================================
# Orthogonalisation Matrix Tests
# ======================================================================


def test_X_shape(h2_scf: _StubRHF) -> None:
    """X should be n_basis × n_basis.

    The orthogonalisation matrix X = S^{-1/2} transforms the
    generalised eigenvalue problem into a standard one.  It
    must be square with dimension equal to the basis size.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # X is the canonical orthogonalisation matrix.
    n = h2_scf.n_basis
    assert h2_scf.X.shape == (n, n)


def test_X_produces_identity(h2_scf: _StubRHF) -> None:
    r"""X^T S X should be the identity matrix.

    This verifies that X diagonalises and normalises the overlap
    matrix, which is the defining property of canonical
    orthogonalisation.  Failure means the S^{-1/2} computation
    is incorrect.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # Product X^T S X must be I if X is a valid orthogonaliser.
    result = h2_scf.X.T @ h2_scf.S @ h2_scf.X
    np.testing.assert_allclose(result, np.eye(h2_scf.n_basis), atol=1e-10)


# ======================================================================
# DIIS and Error Metric Initialisation
# ======================================================================


def test_diis_initialised(h2_scf: _StubRHF) -> None:
    """SCF should initialise a DIIS instance.

    The DIIS accelerator must be created during SCF construction
    so it is ready for use in the first iteration.  Its absence
    would cause an AttributeError during the SCF loop.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert isinstance(h2_scf.diis, DIIS)


def test_error_metric_initialised(h2_scf: _StubRHF) -> None:
    """SCF should initialise a CalculationErrorMetric instance.

    The error metric provides RMS and max-absolute convergence
    measures.  It must be ready before run() starts, because
    it is queried at every iteration for the convergence check.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert isinstance(h2_scf.calculation_error_metric, CalculationErrorMetric)


# ======================================================================
# Result Attributes Before run()
# ======================================================================


def test_converged_none_before_run(h2_scf: _StubRHF) -> None:
    """Before run(), converged should be None.

    The convergence flag is set only after the SCF loop finishes.
    A pre-run value of None distinguishes an uninitialised object
    from one that has been executed.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert h2_scf.converged is None


def test_n_iterations_none_before_run(h2_scf: _StubRHF) -> None:
    """Before run(), n_iterations should be None.

    Iteration count is populated by _collect_results only once
    the SCF loop terminates.  Verifying None avoids false
    positives from stale state carried across test runs.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert h2_scf.n_iterations is None


def test_e_electronic_none_before_run(h2_scf: _StubRHF) -> None:
    """Before run(), e_electronic should be None.

    Electronic energy is computed during the SCF loop and stored
    upon completion.  A None sentinel ensures the attribute is
    not accidentally populated by the constructor.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert h2_scf.e_electronic is None


def test_e_total_none_before_run(h2_scf: _StubRHF) -> None:
    """Before run(), e_total should be None.

    Total energy combines electronic and nuclear-repulsion
    contributions.  It must remain None until the SCF loop
    completes to enforce correct test sequencing.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    assert h2_scf.e_total is None


# ======================================================================
# SCF Run Tests — H2 / STO-3G
# ======================================================================


def test_h2_scf_converges(h2_scf: _StubRHF) -> None:
    """H2 / STO-3G RHF should converge.

    H2 with a minimal basis set is a textbook case that must
    always converge within the default iteration limit.  Failure
    here signals a fundamental problem in the SCF loop.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    assert h2_scf.converged is True


def test_h2_scf_returns_self(h2_scf: _StubRHF) -> None:
    """run() should return self for method chaining.

    Returning self allows patterns like scf.run().e_total in
    a single expression.  The identity check (is) confirms
    the same object is returned, not a copy.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    # result must be the exact same _StubRHF instance.
    result = h2_scf.run()
    assert result is h2_scf


def test_h2_scf_iterations_positive(h2_scf: _StubRHF) -> None:
    """Number of iterations should be a positive integer.

    At least one iteration is required to build the Fock matrix
    and update the density.  A zero count would mean the loop
    body was never executed.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    assert h2_scf.n_iterations is not None
    assert h2_scf.n_iterations > 0


def test_h2_scf_total_energy_reasonable(h2_scf: _StubRHF) -> None:
    """H2 / STO-3G total energy should be around -1.117 Hartree.

    The reference value comes from established quantum-chemistry
    textbooks and other RHF implementations.  The tolerance of
    0.01 Ha accounts for minor numerical differences.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    # Known reference: approximately -1.117 Ha for H2/STO-3G.
    assert h2_scf.e_total is not None
    np.testing.assert_allclose(h2_scf.e_total, -1.117, atol=0.01)


def test_h2_scf_electronic_energy_negative(h2_scf: _StubRHF) -> None:
    """Electronic energy should be negative for a bound system.

    The electronic energy includes electron-nucleus attraction
    and electron-electron repulsion contributions.  For any
    stable molecule this sum must be negative.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    assert h2_scf.e_electronic is not None
    assert h2_scf.e_electronic < 0.0


def test_h2_scf_matrices_populated(h2_scf: _StubRHF) -> None:
    """After run(), matrices dict should contain expected keys.

    _collect_results stores overlap (S), core Hamiltonian (H),
    Fock (F), density (P), MO coefficients (C), and orbital
    energies (epsilon).  All six keys must be present.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    # expected_keys is the full set of required matrix names.
    expected_keys = {"S", "H", "F", "P", "C", "epsilon"}
    assert expected_keys.issubset(h2_scf.matrices.keys())


def test_h2_scf_density_matrix_trace(h2_scf: _StubRHF) -> None:
    """Tr(P S) should equal the number of electrons.

    The trace of the product of the density and overlap matrices
    gives the total electron count.  For H2 this must be exactly
    2.0 (within numerical tolerance).

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    # P is the converged density matrix, S is the overlap matrix.
    P = h2_scf.matrices["P"]
    S = h2_scf.matrices["S"]
    # n_electrons should reproduce the true electron count.
    n_electrons = np.trace(P @ S)
    np.testing.assert_allclose(n_electrons, 2.0, atol=1e-8)


def test_h2_scf_density_idempotent(h2_scf: _StubRHF) -> None:
    r"""For RHF, P S P = P (idempotency in S-metric, scaled by 2).

    Because P = 2 C_occ C_occ^T, the closed-shell density matrix
    satisfies P S P = 2 P.  This is the S-metric idempotency
    relation that guarantees the density is projective.

    :param h2_scf: Pytest fixture providing an unexecuted _StubRHF for H2/STO-3G.
    :type h2_scf: _StubRHF
    """
    h2_scf.run()
    # P is the converged density, S is the AO overlap.
    P = h2_scf.matrices["P"]
    S = h2_scf.matrices["S"]
    # PSP must equal 2P for a valid closed-shell density.
    PSP = P @ S @ P
    np.testing.assert_allclose(PSP, 2.0 * P, atol=1e-8)
