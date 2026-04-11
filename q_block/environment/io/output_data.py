"""Container for quantum chemistry calculation results.

This module provides the :class:`OutputData` class, a structured container
for storing and exporting results from quantum chemistry calculations
(SCF, Hartree-Fock, etc.).

The class mirrors the design philosophy of :class:`InputData`: it provides
a thin, typed wrapper around calculation results, making it easier to pass
structured output data through higher-level APIs and export to various formats.

Design
------
- **Simple container**: stores calculation results without performing
  complex logic.
- **No misleading defaults**: unlike typical dataclasses, numerical fields
  are set to ``None`` by default to avoid confusion about whether a value
  was actually computed or is just a placeholder.
- **Multiple export formats**: JSON (machine-readable), formatted text
  (human-readable), and dictionary (for programmatic access).
- **Factory methods**: construct from SCF solver instances or manually.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from q_block.environment.constants.natural.atoms_data import (
    ATOMS_SYMBOLS_Z_TO_SYMBOL,
)


class OutputData:
    """Structured container for quantum chemistry calculation results.

    This class stores results from quantum chemistry calculations in a
    standardized format. It is designed to be populated either manually
    or via the :meth:`from_scf` factory method.

    All numerical result fields default to ``None`` rather than zero to
    make it clear when values have not been set, avoiding potential
    confusion between a computed zero and an uninitialized field.

    :param method: Calculation method name (e.g., ``"RHF"``, ``"UHF"``).
    :type method: Optional[str]
    :param basis_set: Basis set description.
    :type basis_set: Optional[str]
    :param molecular_formula: Chemical formula.
    :type molecular_formula: Optional[str]
    :param converged: Whether the calculation converged.
    :type converged: Optional[bool]
    :param n_iterations: Number of iterations performed.
    :type n_iterations: Optional[int]
    :param convergence_threshold: Convergence threshold used.
    :type convergence_threshold: Optional[float]
    :param n_electrons: Total number of electrons.
    :type n_electrons: Optional[int]
    :param n_basis: Number of basis functions.
    :type n_basis: Optional[int]
    :param charge: System charge.
    :type charge: Optional[int]
    :param multiplicity: Spin multiplicity (2S+1).
    :type multiplicity: Optional[int]
    :param e_electronic: Electronic energy in Hartree.
    :type e_electronic: Optional[float]
    :param e_nuclear: Nuclear repulsion energy in Hartree.
    :type e_nuclear: Optional[float]
    :param e_total: Total energy in Hartree.
    :type e_total: Optional[float]
    :param orbital_energies: Molecular orbital energies.
    :type orbital_energies: Optional[np.ndarray]
    :param orbital_energies_alpha: Alpha orbital energies (UHF).
    :type orbital_energies_alpha: Optional[np.ndarray]
    :param orbital_energies_beta: Beta orbital energies (UHF).
    :type orbital_energies_beta: Optional[np.ndarray]
    :param matrices: Named matrices (C, P, F, S, epsilon, etc.).
    :type matrices: Optional[Dict[str, np.ndarray]]
    :param extra: Additional calculation-specific data.
    :type extra: Optional[Dict[str, Any]]
    :param geometry: Final molecular geometry.
    :type geometry: Optional[pd.DataFrame]
    :param timestamp: Calculation timestamp (ISO format).
    :type timestamp: Optional[str]
    :param iteration_history: SCF iteration history with energies and errors.
    :type iteration_history: Optional[List[Dict[str, Any]]]
    """

    def __init__(
        self,
        method: Optional[str] = None,
        basis_set: Optional[str] = None,
        molecular_formula: Optional[str] = None,
        converged: Optional[bool] = None,
        n_iterations: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        n_electrons: Optional[int] = None,
        n_basis: Optional[int] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        e_electronic: Optional[float] = None,
        e_nuclear: Optional[float] = None,
        e_total: Optional[float] = None,
        orbital_energies: Optional[np.ndarray] = None,
        orbital_energies_alpha: Optional[np.ndarray] = None,
        orbital_energies_beta: Optional[np.ndarray] = None,
        matrices: Optional[Dict[str, np.ndarray]] = None,
        extra: Optional[Dict[str, Any]] = None,
        geometry: Optional[pd.DataFrame] = None,
        timestamp: Optional[str] = None,
        iteration_history: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Calculation metadata
        self.method: Optional[str] = method
        self.timestamp: str = timestamp or datetime.now().isoformat()
        self.basis_set: Optional[str] = basis_set
        self.molecular_formula: Optional[str] = molecular_formula

        # Convergence information
        self.converged: Optional[bool] = converged
        self.n_iterations: Optional[int] = n_iterations
        self.convergence_threshold: Optional[float] = convergence_threshold

        # System properties
        self.n_electrons: Optional[int] = n_electrons
        self.n_basis: Optional[int] = n_basis
        self.charge: Optional[int] = charge
        self.multiplicity: Optional[int] = multiplicity

        # Energies (Hartree)
        self.e_electronic: Optional[float] = e_electronic
        self.e_nuclear: Optional[float] = e_nuclear
        self.e_total: Optional[float] = e_total

        # Orbital data
        self.orbital_energies: Optional[np.ndarray] = orbital_energies
        self.orbital_energies_alpha: Optional[np.ndarray] = orbital_energies_alpha
        self.orbital_energies_beta: Optional[np.ndarray] = orbital_energies_beta

        # Matrices and extra data
        self.matrices: Dict[str, np.ndarray] = matrices or {}
        self.extra: Dict[str, Any] = extra or {}

        # Geometry
        self.geometry: Optional[pd.DataFrame] = geometry

        # Iteration history
        self.iteration_history: List[Dict[str, Any]] = iteration_history or []

    @classmethod
    def from_scf(
        cls,
        scf_solver,
        method: Optional[str] = None,
        context=None,
        basis_set: Optional[str] = None,
        geometry: Optional[pd.DataFrame] = None,
    ) -> "OutputData":
        """Construct OutputData from an SCF solver instance.

        This factory method extracts results from a completed SCF solver
        and optionally enriches them with context information (electron
        count, charge, multiplicity, molecular formula).

        :param scf_solver: Completed SCF solver instance with results.
        :type scf_solver: SCF
        :param method: Method name (e.g., ``"RHF"``). If ``None``, attempts
            to infer from solver class name.
        :type method: Optional[str]
        :param context: Calculation context with system information.
        :type context: Optional[QuantumCalculationContext]
        :param basis_set: Basis set description.
        :type basis_set: Optional[str]
        :param geometry: Molecular geometry DataFrame.
        :type geometry: Optional[pd.DataFrame]
        :returns: Populated OutputData instance.
        :rtype: OutputData
        """
        # Infer method from class name if not provided
        if method is None and hasattr(scf_solver, "__class__"):
            method = scf_solver.__class__.__name__

        # Extract orbital energies based on what's available in matrices
        orbital_energies = scf_solver.matrices.get("epsilon")
        orbital_energies_alpha = scf_solver.matrices.get("epsilon_alpha")
        orbital_energies_beta = scf_solver.matrices.get("epsilon_beta")

        # Extract context information if provided
        n_electrons = None
        charge = None
        multiplicity = None
        molecular_formula = None

        if context is not None:
            n_electrons = getattr(context, "n_electrons", None)
            charge = getattr(context, "charge", None)
            multiplicity = getattr(context, "multiplicity", None)
            if hasattr(context, "molecule"):
                molecular_formula = getattr(context.molecule, "formula", None)

        # Build geometry DataFrame from molecule atoms if not supplied explicitly
        if geometry is None and context is not None and hasattr(context, "molecule"):
            atoms = getattr(context.molecule, "atoms", [])
            if atoms:
                records = []
                for atom in atoms:
                    z = atom.atomic_number
                    coords = atom.coordinates
                    records.append(
                        {
                            "symbol": ATOMS_SYMBOLS_Z_TO_SYMBOL.get(z, "?"),
                            "Z": z,
                            "x": coords.x,
                            "y": coords.y,
                            "z": coords.z,
                        }
                    )
                geometry = pd.DataFrame(records)

        return cls(
            method=method,
            basis_set=basis_set,
            molecular_formula=molecular_formula,
            converged=getattr(scf_solver, "converged", None),
            n_iterations=getattr(scf_solver, "n_iterations", None),
            convergence_threshold=getattr(scf_solver, "convergence_threshold", None),
            n_electrons=n_electrons,
            n_basis=getattr(scf_solver, "n_basis", None),
            charge=charge,
            multiplicity=multiplicity,
            e_electronic=getattr(scf_solver, "e_electronic", None),
            e_nuclear=getattr(scf_solver, "e_nuclear", None),
            e_total=getattr(scf_solver, "e_total", None),
            orbital_energies=orbital_energies,
            orbital_energies_alpha=orbital_energies_alpha,
            orbital_energies_beta=orbital_energies_beta,
            matrices=(
                scf_solver.matrices.copy()
                if hasattr(scf_solver, "matrices")
                else {}
            ),
            extra=scf_solver.extra.copy() if hasattr(scf_solver, "extra") else {},
            geometry=geometry,
            iteration_history=getattr(scf_solver, "iteration_history", []),
        )

    def to_dict(self, include_matrices: bool = False) -> Dict[str, Any]:
        """Convert to dictionary representation.

        :param include_matrices: Whether to include matrix data. If ``False``,
            only scalar results are included (JSON-serializable).
        :type include_matrices: bool
        :returns: Dictionary with calculation results.
        :rtype: Dict[str, Any]
        """
        data: Dict[str, Any] = {
            "method": self.method,
            "timestamp": self.timestamp,
            "basis_set": self.basis_set,
            "molecular_formula": self.molecular_formula,
        }

        # Convergence data
        if self.converged is not None:
            data["converged"] = self.converged
        if self.n_iterations is not None:
            data["n_iterations"] = self.n_iterations
        if self.convergence_threshold is not None:
            data["convergence_threshold"] = self.convergence_threshold

        # System properties
        if self.n_electrons is not None:
            data["n_electrons"] = self.n_electrons
        if self.n_basis is not None:
            data["n_basis"] = self.n_basis
        if self.charge is not None:
            data["charge"] = self.charge
        if self.multiplicity is not None:
            data["multiplicity"] = self.multiplicity

        # Energies
        energies = {}
        if self.e_electronic is not None:
            energies["electronic"] = self.e_electronic
        if self.e_nuclear is not None:
            energies["nuclear"] = self.e_nuclear
        if self.e_total is not None:
            energies["total"] = self.e_total
        if energies:
            data["energies"] = energies

        # Orbital energies
        if self.orbital_energies is not None:
            data["orbital_energies"] = self.orbital_energies.tolist()
        if self.orbital_energies_alpha is not None:
            data["orbital_energies_alpha"] = self.orbital_energies_alpha.tolist()
        if self.orbital_energies_beta is not None:
            data["orbital_energies_beta"] = self.orbital_energies_beta.tolist()

        # Matrices (optional, can be large)
        if include_matrices and self.matrices:
            data["matrices"] = {k: v.tolist() for k, v in self.matrices.items()}

        # Extra data
        if self.extra:
            data["extra"] = self.extra

        # Geometry
        if self.geometry is not None and not self.geometry.empty:
            data["geometry"] = self.geometry.to_dict(orient="records")

        # Iteration history
        if self.iteration_history:
            data["iteration_history"] = self.iteration_history

        return data

    def to_json(
        self,
        filepath: Union[str, Path],
        include_matrices: bool = False,
        **kwargs: Any,
    ) -> None:
        """Save results to JSON file.

        :param filepath: Output file path.
        :type filepath: Union[str, Path]
        :param include_matrices: Whether to include matrix data.
        :type include_matrices: bool
        :param kwargs: Additional arguments for :func:`json.dump`.
        """
        data = self.to_dict(include_matrices=include_matrices)
        Path(filepath).write_text(
            json.dumps(data, indent=2, **kwargs),
            encoding="utf-8"
        )

    def to_formatted_text(self) -> str:
        """Generate human-readable formatted text output.

        :returns: Formatted calculation summary.
        :rtype: str
        """
        lines = []
        lines.append("=" * 70)
        title = "Quantum Chemistry Calculation Results"
        if self.method:
            title += f" - {self.method}"
        lines.append(title)
        lines.append("=" * 70)
        lines.append(f"  Timestamp          : {self.timestamp}")

        if self.molecular_formula:
            lines.append(f"  Molecular Formula  : {self.molecular_formula}")
        if self.basis_set:
            lines.append(f"  Basis Set          : {self.basis_set}")

        # Input geometry
        if self.geometry is not None and not self.geometry.empty:
            lines.append("")
            lines.append("Input Geometry (Bohr)")
            lines.append("-" * 70)
            header = f"  {'Atom':<6}{'Z':>4}   {'X':>14}  {'Y':>14}  {'Z':>14}"
            lines.append(header)
            for _, row in self.geometry.iterrows():
                lines.append(
                    f"  {row['symbol']:<6}{int(row['Z']):>4}"
                    f"   {row['x']:>14.9f}  {row['y']:>14.9f}  {row['z']:>14.9f}"
                )

        # System information
        if any(
            x is not None
            for x in [self.n_electrons, self.n_basis, self.charge, self.multiplicity]
        ):
            lines.append("")
            lines.append("System Information")
            lines.append("-" * 70)
            if self.n_electrons is not None:
                lines.append(f"  Number of Electrons: {self.n_electrons}")
            if self.n_basis is not None:
                lines.append(f"  Number of Basis Fns: {self.n_basis}")
            if self.charge is not None:
                lines.append(f"  Charge             : {self.charge:+d}")
            if self.multiplicity is not None:
                lines.append(f"  Multiplicity (2S+1): {self.multiplicity}")

        # Convergence
        if any(
            x is not None
            for x in [self.converged, self.n_iterations, self.convergence_threshold]
        ):
            lines.append("")
            lines.append("Convergence")
            lines.append("-" * 70)
            if self.converged is not None:
                lines.append(f"  Converged          : {self.converged}")
            if self.n_iterations is not None:
                lines.append(f"  Iterations         : {self.n_iterations}")
            if self.convergence_threshold is not None:
                lines.append(
                    f"  Threshold          : {self.convergence_threshold:.2e}"
                )

        # Energies
        if any(
            x is not None for x in [self.e_electronic, self.e_nuclear, self.e_total]
        ):
            lines.append("")
            lines.append("Energies (Hartree)")
            lines.append("-" * 70)
            if self.e_electronic is not None:
                lines.append(f"  Electronic Energy  : {self.e_electronic:18.10f}")
            if self.e_nuclear is not None:
                lines.append(f"  Nuclear Repulsion  : {self.e_nuclear:18.10f}")
            if self.e_total is not None:
                lines.append(f"  Total Energy       : {self.e_total:18.10f}")

        # Orbital energies
        if self.orbital_energies is not None:
            lines.append("")
            lines.append("Orbital Energies")
            lines.append("-" * 70)
            lines.append(f"  {self.orbital_energies}")

        if self.orbital_energies_alpha is not None:
            lines.append("")
            lines.append("Alpha Orbital Energies")
            lines.append("-" * 70)
            lines.append(f"  {self.orbital_energies_alpha}")

        if self.orbital_energies_beta is not None:
            lines.append("")
            lines.append("Beta Orbital Energies")
            lines.append("-" * 70)
            lines.append(f"  {self.orbital_energies_beta}")

        # Matrices
        if self.matrices:
            lines.append("")
            lines.append(f"Available Matrices : {sorted(self.matrices.keys())}")

        # Extra data
        if self.extra:
            lines.append("")
            lines.append("Additional Data")
            lines.append("-" * 70)
            for key, value in self.extra.items():
                lines.append(f"  {key}: {value}")

        # Iteration history summary
        if self.iteration_history:
            lines.append("")
            lines.append("Convergence History")
            lines.append("-" * 70)
            lines.append(
                f"  Total iterations: {len(self.iteration_history)}"
            )
            if len(self.iteration_history) > 0:
                first = self.iteration_history[0]
                last = self.iteration_history[-1]
                lines.append(
                    f"  Initial energy  : {first.get('e_electronic', 'N/A'):.10f} Ha"
                )
                lines.append(
                    f"  Final energy    : {last.get('e_electronic', 'N/A'):.10f} Ha"
                )
                lines.append(
                    f"  Final error     : {last.get('error', 'N/A'):.2e}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    def save_text(self, filepath: Union[str, Path]) -> None:
        """Save formatted text output to file.

        :param filepath: Output file path.
        :type filepath: Union[str, Path]
        """
        Path(filepath).write_text(self.to_formatted_text(), encoding="utf-8")

    def __str__(self) -> str:
        """Return formatted text representation.

        :returns: Human-readable calculation summary.
        :rtype: str
        """
        return self.to_formatted_text()

    def __repr__(self) -> str:
        """Return brief representation.

        :returns: Short string describing the OutputData instance.
        :rtype: str
        """
        parts = []
        if self.method:
            parts.append(f"method='{self.method}'")
        if self.converged is not None:
            status = "converged" if self.converged else "NOT converged"
            parts.append(status)
        if self.e_total is not None:
            parts.append(f"E_total={self.e_total:.6f} Ha")
        if self.n_iterations is not None:
            parts.append(f"iterations={self.n_iterations}")

        if parts:
            return f"OutputData({', '.join(parts)})"
        return "OutputData(empty)"
