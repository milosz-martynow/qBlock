r"""Generic Self-Consistent Field (SCF) algorithm for Hartree-Fock methods.

This module provides the SCF framework for Hartree-Fock electronic
structure calculations (RHF, UHF, ROHF).

The SCF procedure solves the nonlinear eigenvalue problem iteratively:

.. math::

    \mathbf{F}[\mathbf{P}]\,\mathbf{C} = \mathbf{S}\,\mathbf{C}\,
        \boldsymbol{\varepsilon}

where the Fock matrix :math:`\mathbf{F}` depends on the density matrix
:math:`\mathbf{P}`, which in turn depends on the MO coefficients
:math:`\mathbf{C}`.  Convergence is reached when :math:`\mathbf{P}`
(and the total energy) no longer change between iterations.

Design
------
The module uses the **template method** pattern.  :class:`SCF` defines
the common SCF skeleton (orthogonalization, diagonalization, convergence
checking, DIIS) and stores the converged results as instance attributes.
Concrete subclasses override a small set of abstract hooks:

- :meth:`_build_fock` — assemble the Fock matrix from the current
  density.
- :meth:`_compute_electronic_energy` — evaluate :math:`E_{elec}` from
  the current density and Fock matrix.
- :meth:`_build_density` — construct :math:`\mathbf{P}` from occupied
  MO coefficients.
- :meth:`_initial_density` — form the initial density guess from the
  MO coefficients obtained by diagonalising :math:`\mathbf{H}`.
- :meth:`_collect_results` — populate result attributes on the
  instance.

Auxiliary algorithms (DIIS, diagonalisation, error measurement) live in
their own modules under ``q_block.methods``.

Classes
-------
SCF
    Abstract base class implementing the SCF loop machinery and
    storing converged results.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import fractional_matrix_power

from q_block.methods.calculation_error_metric import CalculationErrorMetric
from q_block.methods.diagonalisation import diagonalise_fock
from q_block.methods.diis import DIIS
from q_block.theory.basis_functions import ContractedGaussianTypeOrbital
from q_block.theory.integrals.kinetic_energy import KineticEnergy
from q_block.theory.integrals.nuclear_attraction import NuclearAttraction
from q_block.theory.integrals.overlap import Overlap
from q_block.theory.integrals.two_electron_repulsion import TwoElectronRepulsion


# ======================================================================
# S C F   (abstract template)
# ======================================================================
class SCF(ABC):
    r"""Abstract Self-Consistent Field solver for Hartree-Fock methods.

    Implements the common SCF machinery shared by all HF variants
    (RHF, UHF, ROHF).  The algorithm follows the standard sequence:

    1. **Integral computation** — overlap :math:`\mathbf{S}`, kinetic
       energy :math:`\mathbf{T}`, nuclear attraction :math:`\mathbf{V}`,
       two-electron repulsion integrals (ERI).
    2. **SCF initialisation** — orthogonalisation matrix
       :math:`\mathbf{X} = \mathbf{S}^{-1/2}`, initial Fock guess
       :math:`\mathbf{F}_0 = \mathbf{H}`, diagonalisation, and first
       density matrix.
    3. **SCF iteration loop** — build Fock matrix, compute energy and
       DIIS error, check convergence, diagonalise, update density.
    4. **Post-SCF analysis** — delegated to the concrete subclass via
       :meth:`_collect_results`.

    After :meth:`run` completes, converged results are available as
    instance attributes (``converged``, ``n_iterations``,
    ``e_electronic``, ``e_total``, ``matrices``, ``extra``).

    Subclasses must implement the abstract methods listed below.

    :param cgtos: Contracted Gaussian-type orbital basis.
    :type cgtos: List[ContractedGaussianTypeOrbital]
    :param nuclei: List of ``(Z, (x, y, z))`` tuples in Bohr for each
        nucleus.
    :type nuclei: List[Tuple[int, Tuple[float, float, float]]]
    :param e_nuclear: Nuclear repulsion energy in Hartree, typically
        obtained from
        :class:`~q_block.theory.initialization.nuclear_repulsion_energy.NuclearRepulsionEnergy`.
    :type e_nuclear: float
    :param max_iterations: Maximum number of SCF cycles.
    :type max_iterations: int
    :param convergence_threshold: Threshold :math:`\tau` for both
        :math:`|\Delta E|` and RMS of the DIIS error.
    :type convergence_threshold: float
    :param diis_start: SCF iteration at which to begin DIIS
        extrapolation (0-indexed).
    :type diis_start: int
    :param diis_max_vectors: Maximum number of Fock/error pairs stored
        in the DIIS subspace.
    :type diis_max_vectors: int
    :param calculation_error_metric: Error reduction method used for
        convergence checking (default: ``"rms"``).
    :type calculation_error_metric: str

    Attributes
    ----------
    cgtos : List[ContractedGaussianTypeOrbital]
    nuclei : List[Tuple[int, Tuple[float, float, float]]]
    n_basis : int
        Total number of basis functions.
    max_iterations : int
    convergence_threshold : float
    diis_start : int
    S : np.ndarray
        Overlap matrix.
    T : np.ndarray
        Kinetic energy matrix.
    V : np.ndarray
        Nuclear attraction matrix.
    H : np.ndarray
        Core Hamiltonian (:math:`\mathbf{T} + \mathbf{V}`).
    eri : np.ndarray
        Two-electron repulsion integral tensor.
    e_nuclear : float
        Nuclear repulsion energy.
    X : np.ndarray
        Orthogonalisation matrix :math:`\mathbf{S}^{-1/2}`.
    diis : DIIS
        DIIS convergence accelerator instance.
    calculation_error_metric : CalculationErrorMetric
        Error metric instance used for convergence checking.
    converged : Optional[bool]
        Whether the SCF procedure converged (``None`` before
        :meth:`run` is called).
    n_iterations : Optional[int]
        Number of SCF iterations performed.
    e_electronic : Optional[float]
        Final electronic energy in Hartree.
    e_total : Optional[float]
        Total energy (:math:`E_{elec} + E_{nuc}`) in Hartree.
    matrices : Dict[str, np.ndarray]
        Named matrices produced during the calculation (e.g.
        ``"S"``, ``"H"``, ``"F"``, ``"P"``, ``"C"``,
        ``"epsilon"``).  Populated by :meth:`_collect_results`.
    extra : Dict[str, object]
        Arbitrary additional data (e.g. spin contamination
        :math:`\langle\hat{S}^2\rangle`).
    """

    def __init__(
        self,
        cgtos: List[ContractedGaussianTypeOrbital],
        nuclei: List[Tuple[int, Tuple[float, float, float]]],
        e_nuclear: float,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-8,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        calculation_error_metric: str = "rms",
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1; got {max_iterations}."
            )
        if convergence_threshold <= 0.0:
            raise ValueError(
                f"convergence_threshold must be > 0; "
                f"got {convergence_threshold}."
            )

        self.cgtos: List[ContractedGaussianTypeOrbital] = cgtos
        self.nuclei: List[Tuple[int, Tuple[float, float, float]]] = nuclei
        self.max_iterations: int = max_iterations
        self.convergence_threshold: float = convergence_threshold
        self.diis_start: int = diis_start

        # ── Compute one- and two-electron integrals ──────────────────
        overlap = Overlap(cgtos=self.cgtos)
        self.n_basis: int = overlap.n_basis
        self.S: np.ndarray = overlap.matrix

        kinetic = KineticEnergy(cgtos=self.cgtos)
        self.T: np.ndarray = kinetic.matrix

        nuclear = NuclearAttraction(cgtos=self.cgtos, nuclei=self.nuclei)
        self.V: np.ndarray = nuclear.matrix

        self.H: np.ndarray = self.T + self.V

        eri_obj = TwoElectronRepulsion(cgtos=self.cgtos)
        self.eri: np.ndarray = eri_obj.tensor

        # ── Nuclear repulsion energy (precomputed by caller) ─────────
        self.e_nuclear: float = e_nuclear

        # ── Orthogonalisation matrix  X = S^{-1/2} ──────────────────
        self.X: np.ndarray = np.real(
            fractional_matrix_power(self.S, -0.5)
        )

        # ── DIIS accelerator ─────────────────────────────────────────
        self.diis: DIIS = DIIS(max_vectors=diis_max_vectors)

        # ── Error metric ──────────────────────────────────────────────
        self.calculation_error_metric: CalculationErrorMetric = (
            CalculationErrorMetric(metric=calculation_error_metric)
        )

        # ── Result attributes (populated by run()) ───────────────────
        self.converged: Optional[bool] = None
        self.n_iterations: Optional[int] = None
        self.e_electronic: Optional[float] = None
        self.e_total: Optional[float] = None
        self.matrices: Dict[str, np.ndarray] = {}
        self.extra: Dict[str, object] = {}

    # ==================================================================
    # Public interface
    # ==================================================================

    def run(self) -> 'SCF':
        r"""Execute the full SCF procedure.

        Performs SCF initialisation, iteration, convergence checking,
        and result packaging.  Converged data is stored on the
        instance (see class-level attribute documentation).

        :returns: ``self``, with result attributes populated.
        :rtype: SCF
        """
        # ── 1. Initial guess: F₀ = H, diagonalise, build P ──────────
        C, epsilon = diagonalise_fock(self.H, self.X)
        density = self._initial_density(C)

        # ── 2. SCF iteration loop ────────────────────────────────────
        e_electronic_old = 0.0

        for iteration in range(self.max_iterations):
            # 2a. Build Fock matrix from current density
            fock = self._build_fock(density)

            # 2b. Compute electronic energy
            e_electronic = self._compute_electronic_energy(density, fock)

            # 2c. Compute DIIS commutator error
            commutator_error = self.diis.compute_error(fock, density, self.S)

            # 2d. DIIS extrapolation
            if iteration >= self.diis_start:
                fock = self.diis.extrapolate(fock, commutator_error)

            # 2e. Convergence check
            electronic_energy_change = e_electronic - e_electronic_old
            computation_error = self.calculation_error_metric.compute(
                commutator_error
            )

            if (
                abs(electronic_energy_change) < self.convergence_threshold
                and computation_error < self.convergence_threshold
            ):
                C, epsilon = diagonalise_fock(fock, self.X)
                density = self._build_density(C)
                self._collect_results(
                    converged=True,
                    n_iterations=iteration + 1,
                    e_electronic=e_electronic,
                    fock=fock,
                    density=density,
                    C=C,
                    epsilon=epsilon,
                )
                return self

            # 2f. Diagonalise and update density for next iteration
            C, epsilon = diagonalise_fock(fock, self.X)
            density = self._build_density(C)
            e_electronic_old = e_electronic

        # ── 3. Did not converge ──────────────────────────────────────
        self._collect_results(
            converged=False,
            n_iterations=self.max_iterations,
            e_electronic=e_electronic,
            fock=fock,
            density=density,
            C=C,
            epsilon=epsilon,
        )
        return self

    # ==================================================================
    # Abstract hooks (implemented by concrete HF subclasses)
    # ==================================================================

    @abstractmethod
    def _build_fock(
        self, density: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        r"""Build the Fock matrix from the current density.

        For restricted methods this returns a single matrix; for
        unrestricted methods it returns a tuple of matrices (one per
        spin channel).

        :param density: Current density matrix (or tuple of matrices).
        :type density: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :returns: Fock matrix (or tuple of Fock matrices).
        :rtype: Union[np.ndarray, Tuple[np.ndarray, ...]]
        """
        pass

    @abstractmethod
    def _compute_electronic_energy(
        self,
        density: Union[np.ndarray, Tuple[np.ndarray, ...]],
        fock: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None,
    ) -> float:
        r"""Compute the electronic energy from density and Fock matrices.

        .. math::

            E_{elec} = f(\mathbf{P}, \mathbf{F}, \mathbf{H})

        The exact formula depends on the HF spin treatment (restricted
        vs. unrestricted).  Some variants (e.g. ROHF) may not need the
        Fock matrices and can ignore *fock*.

        :param density: Current density matrix (or tuple).
        :type density: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param fock: Current Fock matrix (or tuple).  May be ``None``
            for methods that compute the energy from cached matrices.
        :type fock: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]
        :returns: Electronic energy in Hartree.
        :rtype: float
        """
        pass

    @abstractmethod
    def _build_density(
        self, C: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        r"""Build density matrix from MO coefficients.

        :param C: MO coefficient matrix (or tuple for unrestricted).
        :type C: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :returns: Density matrix (or tuple).
        :rtype: Union[np.ndarray, Tuple[np.ndarray, ...]]
        """
        pass

    @abstractmethod
    def _initial_density(
        self, C: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        r"""Form the initial density guess from core-Hamiltonian MOs.

        Called once after the first diagonalisation of
        :math:`\mathbf{H}`.  Unrestricted methods may duplicate
        :math:`\mathbf{C}` into separate spin channels here.

        :param C: MO coefficients from the core-Hamiltonian guess.
        :type C: np.ndarray
        :returns: Initial density matrix (or tuple).
        :rtype: Union[np.ndarray, Tuple[np.ndarray, ...]]
        """
        pass

    @abstractmethod
    def _collect_results(
        self,
        converged: bool,
        n_iterations: int,
        e_electronic: float,
        fock: Union[np.ndarray, Tuple[np.ndarray, ...]],
        density: Union[np.ndarray, Tuple[np.ndarray, ...]],
        C: Union[np.ndarray, Tuple[np.ndarray, ...]],
        epsilon: Union[np.ndarray, Tuple[np.ndarray, ...]],
    ) -> None:
        """Populate result attributes on the instance.

        Concrete subclasses set ``self.converged``,
        ``self.n_iterations``, ``self.e_electronic``, ``self.e_total``,
        ``self.matrices``, and optionally ``self.extra``.

        :param converged: Whether SCF converged.
        :type converged: bool
        :param n_iterations: Iterations performed.
        :type n_iterations: int
        :param e_electronic: Final electronic energy.
        :type e_electronic: float
        :param fock: Final Fock matrix (or tuple).
        :type fock: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param density: Final density matrix (or tuple).
        :type density: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param C: Final MO coefficients (or tuple).
        :type C: Union[np.ndarray, Tuple[np.ndarray, ...]]
        :param epsilon: Final orbital energies (or tuple).
        :type epsilon: Union[np.ndarray, Tuple[np.ndarray, ...]]
        """
        pass

    def __repr__(self) -> str:
        if self.converged is None:
            status = "not yet run"
        elif self.converged:
            status = f"converged, e_total={self.e_total:.10f}"
        else:
            status = "not converged"
        return (
            f"{type(self).__name__}(n_basis={self.n_basis}, "
            f"max_iter={self.max_iterations}, "
            f"{status})"
        )
