"""Quantum calculation context classes for quantum-chemistry calculations.

This module implements a multi-level hierarchy:

1. :class:`QuantumCalculationContext` – generic parent class that every
   quantum-chemistry calculation type needs.  It handles:

   * Loading molecular geometry (from an XYZ file *or* a Python
     structure)
   * Reading charge and spin multiplicity
   * Building a :class:`~compute.models.molecule.Molecule`
   * Computing the total electron count (accounting for charge)
   * Converting atomic coordinates from Ångström to Bohr (via
     :meth:`Molecule.to_bohr`)
   * Building and caching the CGTO basis set lazily
   * Computing the total number of basis functions

2. Shared electron-count mixins:

   * :class:`RestrictedContext` – validates even electrons and singlet
     multiplicity; sets :math:`N_{occ} = N_{elec}/2`, :math:`N_\\alpha
     = N_\\beta = N_{occ}`.
   * :class:`UnrestrictedContext` – computes :math:`N_\\alpha`,
     :math:`N_\\beta` from the multiplicity for any spin state.

3. :class:`HartreeFock` – Hartree-Fock base class:

   * :class:`RHF` – Restricted Closed-Shell (via
     :class:`RestrictedContext`).
   * :class:`UHF` – Unrestricted (via :class:`UnrestrictedContext`).
   * :class:`ROHF` – Restricted Open-Shell.

4. :class:`DensityFunctionalTheory` – Kohn-Sham DFT base class:

   * :class:`RKS` – Restricted Closed-Shell KS (via
     :class:`RestrictedContext`).
   * :class:`UKS` – Unrestricted KS (via
     :class:`UnrestrictedContext`).
"""

from typing import List, Literal, Optional

from q_block.compute.models.basis_functions import (
    ContractedGaussianTypeOrbital,
)
from q_block.compute.models.molecule import Molecule

# ──────────────────────────────────────────────────────────────────────
# Allowed method literals
# ──────────────────────────────────────────────────────────────────────
HFMethod = Literal["RHF", "UHF", "ROHF"]
KSMethod = Literal["RKS", "UKS", "ROKS"]


# ======================================================================
# Q U A N T U M   C A L C U L A T I O N   C O N T E X T   (parent)
# ======================================================================
class QuantumCalculationContext:
    """Generic context common to every quantum-chemistry calculation.

    This class encapsulates the first, method-agnostic, steps that
    are always required:

    1. Load molecular geometry into a :class:`Molecule`.
    2. Read charge (sum of per-atom charges) and multiplicity from
       the molecule.
    3. Compute the total number of electrons.
    4. Convert atomic coordinates from Ångström to Bohr.
    5. Compute the total number of basis functions.
    6. Build and cache the CGTO basis set lazily.

    Validation of the electron count and multiplicity consistency is
    performed automatically inside :class:`Molecule` at construction
    time.

    The ``charge`` is derived automatically from per-atom
    :attr:`~compute.models.atom.Atom.charge` values, and
    ``multiplicity`` is set on the
    :class:`~compute.models.molecule.Molecule` itself.

    :param molecule: A :class:`Molecule` instance with geometry
        already loaded (atoms, coordinates, basis set assignments,
        per-atom charges, multiplicity, etc.).
    :type molecule: Molecule

    Attributes
    ----------
    molecule : Molecule
        The molecular system.
    charge : int
        System charge (sum of per-atom charges, read from the
        molecule).
    multiplicity : int
        Spin multiplicity :math:`2S+1` (read from the molecule).
    n_electrons : int
        Total electron count, :math:`\\sum Z_i + \\text{charge}`.
    n_basis : int
        Total number of basis functions (contracted Gaussians) in
        the molecule.
    """

    def __init__(
        self,
        molecule: Molecule,
    ) -> None:
        # ── 1. Store molecule ────────────────────────────────────
        self.molecule: Molecule = molecule

        # ── 2. Read charge and multiplicity from the molecule ────
        self.charge: int = self.molecule.charge
        self.multiplicity: int = self.molecule.multiplicity

        # ── 3. Compute total electron count ──────────────────────
        self.n_electrons: int = self.molecule.n_electrons

        # ── 4. Convert coordinates to Bohr ───────────────────────
        self.molecule.to_bohr()

        # ── 5. Compute basis-set size ────────────────────────────
        self.n_basis: int = self.molecule.n_basis

        # ── 6. Lazy CGTO list (built on first access) ───────────
        self._cgto: Optional[
            List[ContractedGaussianTypeOrbital]
        ] = None

    @property
    def cgto(self) -> List[ContractedGaussianTypeOrbital]:
        """Contracted Gaussian-Type Orbital basis.

        Built lazily on first access by calling
        :meth:`Molecule.make_contracted_gaussian_type_orbital`.
        Subsequent accesses return the cached list.

        :returns: Flat list of
            :class:`~compute.models.basis_functions.ContractedGaussianTypeOrbital`
            shells.
        :rtype: list
        """
        if self._cgto is None:
            self.molecule.make_contracted_gaussian_type_orbital()
            self._cgto = (
                self.molecule.contracted_gaussian_type_orbitals
                or []
            )
        return self._cgto

    def __repr__(self) -> str:
        return (
            f"QuantumCalculationContext("
            f"n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons})"
        )


# ======================================================================
# R E S T R I C T E D / U N R E S T R I C T E D   M I X I N S
# ======================================================================
class RestrictedContext:
    r"""Mixin: restricted closed-shell electron-count logic.

    Validates that the system has an **even** number of electrons
    and **singlet** multiplicity (:math:`M = 1`), then computes:

    .. math:: N_{occ} = N_{elec} / 2

    Both HF (:class:`RHF`) and KS-DFT (:class:`RKS`) restricted
    methods inherit from this mixin.

    Attributes
    ----------
    n_occ : int
        Number of doubly-occupied spatial orbitals.
    n_alpha : int
        Number of alpha electrons (equal to :attr:`n_occ`).
    n_beta : int
        Number of beta electrons (equal to :attr:`n_occ`).
    """

    n_electrons: int
    charge: int
    multiplicity: int

    def _init_restricted(self) -> None:
        """Validate and set restricted electron counts."""
        if self.n_electrons % 2 != 0:
            raise ValueError(
                f"Restricted calculation requires an even "
                f"number of electrons; got "
                f"{self.n_electrons} "
                f"(charge={self.charge}, "
                f"multiplicity={self.multiplicity})."
            )
        if self.multiplicity != 1:
            raise ValueError(
                f"Restricted calculation requires singlet "
                f"multiplicity (1); got "
                f"{self.multiplicity}."
            )

        self.n_occ: int = self.n_electrons // 2
        self.n_alpha: int = self.n_occ
        self.n_beta: int = self.n_occ


class UnrestrictedContext:
    r"""Mixin: unrestricted electron-count logic.

    Computes alpha and beta electron counts from the multiplicity:

    .. math::

        N_{\beta}  = (N_{elec} - n_{unpaired}) / 2

        N_{\alpha} = N_{\beta} + n_{unpaired}

    where :math:`n_{unpaired} = M - 1`.

    Both HF (:class:`UHF`) and KS-DFT (:class:`UKS`) unrestricted
    methods inherit from this mixin.

    Attributes
    ----------
    n_alpha : int
        Number of alpha (spin-up) electrons.
    n_beta : int
        Number of beta (spin-down) electrons.
    """

    n_electrons: int
    multiplicity: int

    def _init_unrestricted(self) -> None:
        """Compute and set unrestricted electron counts."""
        n_unpaired = self.multiplicity - 1
        self.n_beta: int = (
            self.n_electrons - n_unpaired
        ) // 2
        self.n_alpha: int = self.n_beta + n_unpaired


# ======================================================================
# H A R T R E E – F O C K   B A S E   C L A S S
# ======================================================================
class HartreeFock(QuantumCalculationContext):
    """Hartree-Fock base class with logic common to all HF variants.

    Inherits all generic setup from
    :class:`QuantumCalculationContext` and additionally stores
    the HF method label and declares the common electron-count
    attributes :attr:`n_alpha` and :attr:`n_beta` (populated by
    the concrete subclass).

    Concrete subclasses (:class:`RHF`, :class:`UHF`, :class:`ROHF`)
    implement method-specific electron-count logic and validation.

    :param molecule: Molecular system with basis-set data and
        per-atom charges already attached to each atom, and
        ``multiplicity`` set on the molecule.
    :type molecule: Molecule
    :param hf_method: Hartree-Fock variant label (``"RHF"``,
        ``"UHF"``, or ``"ROHF"``).
    :type hf_method: HFMethod

    Attributes
    ----------
    hf_method : HFMethod
        Validated HF method string.
    n_alpha : Optional[int]
        Number of alpha electrons (set by subclass).
    n_beta : Optional[int]
        Number of beta electrons (set by subclass).
    """

    def __init__(
        self,
        molecule: Molecule,
        hf_method: HFMethod,
    ) -> None:
        super().__init__(molecule=molecule)

        self.hf_method: HFMethod = hf_method

        self.n_alpha: Optional[int] = None
        self.n_beta: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"hf_method={self.hf_method}, "
            f"n_alpha={self.n_alpha}, "
            f"n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R H F   –   R E S T R I C T E D   C L O S E D - S H E L L
# ======================================================================
class RHF(HartreeFock, RestrictedContext):
    """Restricted Closed-Shell Hartree-Fock.

    Requires an **even** number of electrons and **singlet**
    multiplicity (:math:`M = 1`).  All electrons are paired:

    .. math:: N_{occ} = N_{elec} / 2

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule

    Attributes
    ----------
    n_occ : int
        Number of doubly-occupied spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="RHF")
        self._init_restricted()

    def __repr__(self) -> str:
        return (
            f"RHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_occ={self.n_occ}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# U H F   –   U N R E S T R I C T E D
# ======================================================================
class UHF(HartreeFock, UnrestrictedContext):
    """Unrestricted Hartree-Fock.

    Uses separate spatial orbitals for alpha and beta electrons,
    supporting any spin multiplicity:

    .. math::

        N_{\\beta}  = (N_{elec} - n_{unpaired}) / 2

        N_{\\alpha} = N_{\\beta} + n_{unpaired}

    where :math:`n_{unpaired} = M - 1`.

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="UHF")
        self._init_unrestricted()

    def __repr__(self) -> str:
        return (
            f"UHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_alpha={self.n_alpha}, "
            f"n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R O H F   –   R E S T R I C T E D   O P E N - S H E L L
# ======================================================================
class ROHF(HartreeFock):
    """Restricted Open-Shell Hartree-Fock.

    Doubly-occupied (closed) orbitals share the same spatial part;
    singly-occupied (open) orbitals carry unpaired electrons:

    .. math::

        N_{open}   = M - 1

        N_{closed} = (N_{elec} - N_{open}) / 2

    Verifies :math:`M = N_{open} + 1`.

    :param molecule: Molecular system (see :class:`HartreeFock`).
    :type molecule: Molecule

    Attributes
    ----------
    n_closed : int
        Number of doubly-occupied (closed-shell) spatial orbitals.
    n_open : int
        Number of singly-occupied (open-shell) spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, hf_method="ROHF")

        # ── Validate and compute ─────────────────────────────────
        n_unpaired = self.multiplicity - 1
        n_paired_electrons = self.n_electrons - n_unpaired
        if n_paired_electrons % 2 != 0:
            raise ValueError(
                f"ROHF: after removing {n_unpaired} unpaired "
                f"electrons, the remaining "
                f"{n_paired_electrons} electrons are not even."
            )

        self.n_closed: int = n_paired_electrons // 2
        self.n_open: int = n_unpaired

        self.n_beta = self.n_closed
        self.n_alpha = self.n_closed + self.n_open

        # Verify multiplicity = N_open + 1
        expected_multiplicity = self.n_open + 1
        if self.multiplicity != expected_multiplicity:
            raise ValueError(
                f"ROHF multiplicity inconsistency: "
                f"multiplicity={self.multiplicity} but "
                f"N_open={self.n_open} implies "
                f"M={expected_multiplicity}."
            )

    def __repr__(self) -> str:
        return (
            f"ROHF(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_closed={self.n_closed}, "
            f"n_open={self.n_open}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# D E N S I T Y   F U N C T I O N A L   T H E O R Y   B A S E
# ======================================================================
class DensityFunctionalTheory(QuantumCalculationContext):
    """Base class for all Kohn-Sham DFT calculation contexts.

    Inherits generic setup from :class:`QuantumCalculationContext`
    and additionally stores the KS method label and declares common
    electron-count attributes :attr:`n_alpha` and :attr:`n_beta`
    (populated by the concrete subclass).

    :param molecule: Molecular system with basis-set data and
        per-atom charges already attached, and ``multiplicity``
        set.
    :type molecule: Molecule
    :param ks_method: Kohn-Sham variant label (``"RKS"`` or
        ``"UKS"``).
    :type ks_method: KSMethod

    Attributes
    ----------
    ks_method : KSMethod
        Validated KS method string.
    n_alpha : Optional[int]
        Number of alpha electrons (set by subclass).
    n_beta : Optional[int]
        Number of beta electrons (set by subclass).
    """

    def __init__(
        self,
        molecule: Molecule,
        ks_method: KSMethod,
    ) -> None:
        super().__init__(molecule=molecule)

        self.ks_method: KSMethod = ks_method

        self.n_alpha: Optional[int] = None
        self.n_beta: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"ks_method={self.ks_method}, "
            f"n_alpha={self.n_alpha}, "
            f"n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R K S   –   R E S T R I C T E D   C L O S E D - S H E L L
# ======================================================================
class RKS(DensityFunctionalTheory, RestrictedContext):
    """Restricted Closed-Shell Kohn-Sham DFT.

    Requires an **even** number of electrons and **singlet**
    multiplicity (:math:`M = 1`).  All electrons are paired:

    .. math:: N_{occ} = N_{elec} / 2

    :param molecule: Molecular system (see
        :class:`DensityFunctionalTheory`).
    :type molecule: Molecule

    Attributes
    ----------
    n_occ : int
        Number of doubly-occupied spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, ks_method="RKS")
        self._init_restricted()

    def __repr__(self) -> str:
        return (
            f"RKS(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_occ={self.n_occ}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# U K S   –   U N R E S T R I C T E D
# ======================================================================
class UKS(DensityFunctionalTheory, UnrestrictedContext):
    """Unrestricted Kohn-Sham DFT.

    Uses separate spatial orbitals for alpha and beta electrons,
    supporting any spin multiplicity:

    .. math::

        N_{\\beta}  = (N_{elec} - n_{unpaired}) / 2

        N_{\\alpha} = N_{\\beta} + n_{unpaired}

    where :math:`n_{unpaired} = M - 1`.

    :param molecule: Molecular system (see
        :class:`DensityFunctionalTheory`).
    :type molecule: Molecule
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(
            molecule=molecule, ks_method="UKS"
        )
        self._init_unrestricted()

    def __repr__(self) -> str:
        return (
            f"UKS(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_alpha={self.n_alpha}, "
            f"n_beta={self.n_beta}, "
            f"n_basis={self.n_basis})"
        )


# ======================================================================
# R O K S   –   R E S T R I C T E D   O P E N - S H E L L   K S
# ======================================================================
class ROKS(DensityFunctionalTheory):
    """Restricted Open-Shell Kohn-Sham DFT.

    Mirrors :class:`ROHF` for Kohn-Sham DFT.  Doubly-occupied
    (closed) orbitals share the same spatial part; singly-occupied
    (open) orbitals carry unpaired electrons:

    .. math::

        N_{open}   = M - 1

        N_{closed} = (N_{elec} - N_{open}) / 2

    :param molecule: Molecular system (see
        :class:`DensityFunctionalTheory`).
    :type molecule: Molecule

    Attributes
    ----------
    n_closed : int
        Number of doubly-occupied (closed-shell) spatial orbitals.
    n_open : int
        Number of singly-occupied (open-shell) spatial orbitals.
    """

    def __init__(self, molecule: Molecule) -> None:
        super().__init__(molecule=molecule, ks_method="ROKS")

        n_unpaired = self.multiplicity - 1
        n_paired_electrons = self.n_electrons - n_unpaired
        if n_paired_electrons % 2 != 0:
            raise ValueError(
                f"ROKS: after removing {n_unpaired} unpaired "
                f"electrons, the remaining "
                f"{n_paired_electrons} electrons are not even."
            )

        self.n_closed: int = n_paired_electrons // 2
        self.n_open: int = n_unpaired

        self.n_beta = self.n_closed
        self.n_alpha = self.n_closed + self.n_open

        expected_multiplicity = self.n_open + 1
        if self.multiplicity != expected_multiplicity:
            raise ValueError(
                f"ROKS multiplicity inconsistency: "
                f"multiplicity={self.multiplicity} but "
                f"N_open={self.n_open} implies "
                f"M={expected_multiplicity}."
            )

    def __repr__(self) -> str:
        return (
            f"ROKS(n_atoms={len(self.molecule)}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"n_electrons={self.n_electrons}, "
            f"n_closed={self.n_closed}, "
            f"n_open={self.n_open}, "
            f"n_basis={self.n_basis})"
        )
