"""Unit tests for q_block.io.output_data module.

Tests cover:
- OutputData.__init__: manual construction with all parameters
- OutputData.from_scf: factory method from SCF solver instances
- OutputData.to_dict: dictionary conversion with/without matrices
- OutputData.to_json: JSON file export
- OutputData.to_formatted_text: human-readable text generation
- OutputData.save_text: text file export
- Iteration history storage and retrieval

All tests use pytest with parametrize, no test classes.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from q_block.environment.io.output_data import OutputData

# ======================================================================
# OutputData.__init__ Tests
# ======================================================================


def test_output_data_init_empty() -> None:
    """Verify OutputData can be initialized empty with all defaults None.

    All numerical fields should default to None, not zero.
    """
    output: OutputData = OutputData()

    assert output.method is None
    assert output.basis_set is None
    assert output.molecular_formula is None
    assert output.converged is None
    assert output.n_iterations is None
    assert output.convergence_threshold is None
    assert output.n_electrons is None
    assert output.n_basis is None
    assert output.charge is None
    assert output.multiplicity is None
    assert output.e_electronic is None
    assert output.e_nuclear is None
    assert output.e_total is None
    assert output.orbital_energies is None
    assert output.orbital_energies_alpha is None
    assert output.orbital_energies_beta is None
    assert output.matrices == {}
    assert output.extra == {}
    assert output.geometry is None
    assert output.iteration_history == []
    assert isinstance(output.timestamp, str)
    assert len(output.timestamp) > 0


def test_output_data_init_with_all_parameters() -> None:
    """Verify OutputData stores all provided parameters correctly."""
    test_matrices: Dict[str, np.ndarray] = {
        "C": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "F": np.array([[0.5, 0.1], [0.1, 0.5]]),
    }
    test_extra: Dict[str, object] = {"spin_contamination": 0.0}
    test_geometry: pd.DataFrame = pd.DataFrame(
        {"atom_id": ["A1", "A2"], "symbol": ["H", "H"]}
    )
    test_iterations: List[Dict[str, float]] = [
        {"iteration": 1, "e_electronic": -1.0, "energy_change": -1.0, "error": 0.1},
        {
            "iteration": 2,
            "e_electronic": -1.5,
            "energy_change": -0.5,
            "error": 0.01,
        },
    ]

    output: OutputData = OutputData(
        method="RHF",
        basis_set="STO-3G",
        molecular_formula="H2",
        converged=True,
        n_iterations=10,
        convergence_threshold=1e-8,
        n_electrons=2,
        n_basis=2,
        charge=0,
        multiplicity=1,
        e_electronic=-1.83,
        e_nuclear=0.71,
        e_total=-1.12,
        orbital_energies=np.array([-0.59, 0.26]),
        matrices=test_matrices,
        extra=test_extra,
        geometry=test_geometry,
        timestamp="2026-04-09T12:00:00",
        iteration_history=test_iterations,
    )

    assert output.method == "RHF"
    assert output.basis_set == "STO-3G"
    assert output.molecular_formula == "H2"
    assert output.converged is True
    assert output.n_iterations == 10
    assert output.convergence_threshold == 1e-8
    assert output.n_electrons == 2
    assert output.n_basis == 2
    assert output.charge == 0
    assert output.multiplicity == 1
    assert output.e_electronic == pytest.approx(-1.83)
    assert output.e_nuclear == pytest.approx(0.71)
    assert output.e_total == pytest.approx(-1.12)
    assert np.allclose(output.orbital_energies, np.array([-0.59, 0.26]))
    assert "C" in output.matrices
    assert "F" in output.matrices
    assert output.extra["spin_contamination"] == 0.0
    assert len(output.geometry) == 2
    assert output.timestamp == "2026-04-09T12:00:00"
    assert len(output.iteration_history) == 2
    assert output.iteration_history[0]["iteration"] == 1


@pytest.mark.parametrize(
    "converged, n_iterations, expected_repr",
    [
        (True, 5, "converged"),
        (False, 100, "NOT converged"),
        (None, None, "OutputData(method='TEST')"),
    ],
    ids=["converged", "not_converged", "empty_status"],
)
def test_output_data_repr(
    converged: Optional[bool], n_iterations: Optional[int], expected_repr: str
) -> None:
    """Verify __repr__ shows appropriate convergence status.

    :param converged: Convergence status.
    :type converged: Optional[bool]
    :param n_iterations: Number of iterations.
    :type n_iterations: Optional[int]
    :param expected_repr: Expected substring in repr.
    :type expected_repr: str
    """
    output: OutputData = OutputData(
        method="TEST",
        converged=converged,
        n_iterations=n_iterations,
    )
    repr_str: str = repr(output)
    assert expected_repr in repr_str


# ======================================================================
# OutputData.from_scf Tests
# ======================================================================


def test_output_data_from_scf_mock_solver() -> None:
    """Verify from_scf extracts data from a mock SCF solver."""

    class MockSCFSolver:
        """Mock SCF solver for testing."""

        def __init__(self) -> None:
            """Initialize mock solver."""
            self.converged: bool = True
            self.n_iterations: int = 5
            self.convergence_threshold: float = 1e-8
            self.n_basis: int = 4
            self.e_electronic: float = -1.838
            self.e_nuclear: float = 0.715
            self.e_total: float = -1.123
            self.matrices: Dict[str, np.ndarray] = {
                "epsilon": np.array([-0.59, 0.26]),
            }
            self.extra: Dict[str, object] = {}
            self.iteration_history: List[Dict[str, float]] = [
                {
                    "iteration": 1,
                    "e_electronic": -1.788,
                    "energy_change": -1.788,
                    "error": 0.071,
                    "diis_active": False,
                },
                {
                    "iteration": 2,
                    "e_electronic": -1.838,
                    "energy_change": -0.050,
                    "error": 0.011,
                    "diis_active": True,
                },
            ]

    class MockContext:
        """Mock context for testing."""

        def __init__(self) -> None:
            """Initialize mock context."""
            self.n_electrons: int = 2
            self.charge: int = 0
            self.multiplicity: int = 1

            class MockMolecule:
                """Mock molecule."""

                formula: str = "H2"

            self.molecule: MockMolecule = MockMolecule()

    solver: MockSCFSolver = MockSCFSolver()
    context: MockContext = MockContext()

    output: OutputData = OutputData.from_scf(
        scf_solver=solver,
        method="RHF",
        context=context,
        basis_set="3-21G",
    )

    assert output.method == "RHF"
    assert output.basis_set == "3-21G"
    assert output.converged is True
    assert output.n_iterations == 5
    assert output.convergence_threshold == pytest.approx(1e-8)
    assert output.n_electrons == 2
    assert output.n_basis == 4
    assert output.charge == 0
    assert output.multiplicity == 1
    assert output.e_electronic == pytest.approx(-1.838)
    assert output.e_nuclear == pytest.approx(0.715)
    assert output.e_total == pytest.approx(-1.123)
    assert output.orbital_energies is not None
    assert len(output.iteration_history) == 2
    assert output.iteration_history[0]["iteration"] == 1
    assert output.iteration_history[1]["diis_active"] is True


def test_output_data_from_scf_infers_method_from_class() -> None:
    """Verify from_scf infers method name from solver class if not provided."""

    class RestrictedHartreeFock:
        """Mock RHF solver."""

        def __init__(self) -> None:
            """Initialize mock solver."""
            self.converged: bool = True
            self.matrices: Dict[str, np.ndarray] = {}
            self.extra: Dict[str, object] = {}

    solver: RestrictedHartreeFock = RestrictedHartreeFock()
    output: OutputData = OutputData.from_scf(scf_solver=solver)

    assert output.method == "RestrictedHartreeFock"


# ======================================================================
# OutputData.to_dict Tests
# ======================================================================


def test_output_data_to_dict_excludes_none_values() -> None:
    """Verify to_dict excludes fields that are None."""
    output: OutputData = OutputData(
        method="RHF",
        converged=True,
        e_total=-1.12,
    )

    data: Dict[str, object] = output.to_dict(include_matrices=False)

    assert "method" in data
    assert "converged" in data
    assert "energies" in data
    assert data["energies"]["total"] == pytest.approx(-1.12)
    assert "n_electrons" not in data
    assert "n_basis" not in data


def test_output_data_to_dict_includes_iteration_history() -> None:
    """Verify to_dict includes iteration history when present."""
    iterations: List[Dict[str, float]] = [
        {"iteration": 1, "e_electronic": -1.0, "error": 0.1},
        {"iteration": 2, "e_electronic": -1.5, "error": 0.01},
    ]
    output: OutputData = OutputData(iteration_history=iterations)

    data: Dict[str, object] = output.to_dict()

    assert "iteration_history" in data
    assert len(data["iteration_history"]) == 2
    assert data["iteration_history"][0]["iteration"] == 1


@pytest.mark.parametrize(
    "include_matrices, expected_in_dict",
    [
        (False, False),
        (True, True),
    ],
    ids=["exclude_matrices", "include_matrices"],
)
def test_output_data_to_dict_matrices_optional(
    include_matrices: bool, expected_in_dict: bool
) -> None:
    """Verify to_dict respects include_matrices parameter.

    :param include_matrices: Whether to include matrices in output.
    :type include_matrices: bool
    :param expected_in_dict: Whether matrices should be in result.
    :type expected_in_dict: bool
    """
    output: OutputData = OutputData(
        matrices={"C": np.array([[1.0, 0.0], [0.0, 1.0]])}
    )

    data: Dict[str, object] = output.to_dict(include_matrices=include_matrices)

    assert ("matrices" in data) == expected_in_dict


def test_output_data_to_dict_orbital_energies_converted_to_list() -> None:
    """Verify to_dict converts numpy arrays to lists for JSON compatibility."""
    output: OutputData = OutputData(
        orbital_energies=np.array([-0.59, 0.26]),
        orbital_energies_alpha=np.array([-0.60, 0.25]),
        orbital_energies_beta=np.array([-0.58, 0.27]),
    )

    data: Dict[str, object] = output.to_dict()

    assert isinstance(data["orbital_energies"], list)
    assert isinstance(data["orbital_energies_alpha"], list)
    assert isinstance(data["orbital_energies_beta"], list)
    assert data["orbital_energies"][0] == pytest.approx(-0.59)


# ======================================================================
# OutputData.to_json Tests
# ======================================================================


def test_output_data_to_json_creates_file(tmp_path: Path) -> None:
    """Verify to_json creates a valid JSON file.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    output: OutputData = OutputData(
        method="RHF",
        e_total=-1.12,
        converged=True,
        iteration_history=[
            {"iteration": 1, "e_electronic": -1.0},
        ],
    )

    json_file: Path = tmp_path / "test_output.json"
    output.to_json(json_file, include_matrices=False)

    assert json_file.exists()
    content: str = json_file.read_text(encoding="utf-8")
    assert "RHF" in content
    assert "iteration_history" in content
    assert "-1.12" in content


def test_output_data_to_json_with_matrices(tmp_path: Path) -> None:
    """Verify to_json can include matrices when requested.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    output: OutputData = OutputData(
        matrices={"C": np.array([[1.0, 0.0], [0.0, 1.0]])}
    )

    json_file: Path = tmp_path / "test_with_matrices.json"
    output.to_json(json_file, include_matrices=True)

    content: str = json_file.read_text(encoding="utf-8")
    assert "matrices" in content
    assert '"C"' in content


# ======================================================================
# OutputData.to_formatted_text Tests
# ======================================================================


def test_output_data_to_formatted_text_structure() -> None:
    """Verify to_formatted_text generates properly formatted output."""
    output: OutputData = OutputData(
        method="RHF",
        basis_set="STO-3G",
        n_electrons=2,
        n_basis=2,
        charge=0,
        multiplicity=1,
        converged=True,
        n_iterations=5,
        e_electronic=-1.83,
        e_nuclear=0.71,
        e_total=-1.12,
    )

    text: str = output.to_formatted_text()

    assert "Quantum Chemistry Calculation Results - RHF" in text
    assert "STO-3G" in text
    assert "System Information" in text
    assert "Number of Electrons: 2" in text
    assert "Convergence" in text
    assert "Converged          : True" in text
    assert "Energies (Hartree)" in text
    assert "-1.8300000000" in text
    assert "=" * 70 in text


def test_output_data_to_formatted_text_with_iteration_history() -> None:
    """Verify to_formatted_text includes convergence history summary."""
    output: OutputData = OutputData(
        method="RHF",
        iteration_history=[
            {"iteration": 1, "e_electronic": -1.788, "error": 0.071},
            {"iteration": 2, "e_electronic": -1.838, "error": 0.011},
            {"iteration": 3, "e_electronic": -1.838, "error": 5.9e-10},
        ],
    )

    text: str = output.to_formatted_text()

    assert "Convergence History" in text
    assert "Total iterations: 3" in text
    assert "Initial energy  : -1.7880000000 Ha" in text
    assert "Final energy    : -1.8380000000 Ha" in text
    assert "5.90e-10" in text


def test_output_data_to_formatted_text_omits_empty_sections() -> None:
    """Verify to_formatted_text omits sections when data is None."""
    output: OutputData = OutputData(method="RHF")

    text: str = output.to_formatted_text()

    assert "System Information" not in text
    assert "Convergence" not in text
    assert "Energies" not in text
    assert "Convergence History" not in text


# ======================================================================
# OutputData.save_text Tests
# ======================================================================


def test_output_data_save_text_creates_file(tmp_path: Path) -> None:
    """Verify save_text creates a text file with formatted output.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    output: OutputData = OutputData(
        method="RHF",
        e_total=-1.12,
        converged=True,
    )

    text_file: Path = tmp_path / "test_output.txt"
    output.save_text(text_file)

    assert text_file.exists()
    content: str = text_file.read_text(encoding="utf-8")
    assert "RHF" in content
    assert "-1.12" in content
    assert "Converged          : True" in content


# ======================================================================
# OutputData.__str__ Tests
# ======================================================================


def test_output_data_str_returns_formatted_text() -> None:
    """Verify __str__ returns the same output as to_formatted_text."""
    output: OutputData = OutputData(method="UHF", converged=False)

    str_output: str = str(output)
    formatted: str = output.to_formatted_text()

    assert str_output == formatted
    assert "UHF" in str_output


# ======================================================================
# Iteration History Tests
# ======================================================================


def test_output_data_iteration_history_empty_by_default() -> None:
    """Verify iteration_history is an empty list by default."""
    output: OutputData = OutputData()
    assert output.iteration_history == []
    assert isinstance(output.iteration_history, list)


def test_output_data_iteration_history_preserved() -> None:
    """Verify iteration history is stored and retrieved correctly."""
    iterations: List[Dict[str, float]] = [
        {
            "iteration": 1,
            "e_electronic": -1.788,
            "energy_change": -1.788,
            "error": 0.071,
            "diis_active": False,
        },
        {
            "iteration": 2,
            "e_electronic": -1.837,
            "energy_change": -0.049,
            "error": 0.011,
            "diis_active": True,
        },
        {
            "iteration": 3,
            "e_electronic": -1.838,
            "energy_change": -0.001,
            "error": 0.002,
            "diis_active": True,
        },
    ]

    output: OutputData = OutputData(iteration_history=iterations)

    assert len(output.iteration_history) == 3
    assert output.iteration_history[0]["iteration"] == 1
    assert output.iteration_history[0]["diis_active"] is False
    assert output.iteration_history[1]["diis_active"] is True
    assert output.iteration_history[2]["e_electronic"] == pytest.approx(-1.838)


# ======================================================================
# Edge Cases and Error Handling
# ======================================================================


def test_output_data_handles_zero_values_not_none() -> None:
    """Verify OutputData distinguishes between 0 and None.

    A charge of 0 should be stored and exported, not treated as None.
    """
    output: OutputData = OutputData(
        charge=0,
        e_electronic=0.0,
        e_nuclear=0.0,
    )

    assert output.charge == 0
    assert output.e_electronic == pytest.approx(0.0)
    assert output.e_nuclear == pytest.approx(0.0)

    data: Dict[str, object] = output.to_dict()
    assert data["charge"] == 0
    assert data["energies"]["electronic"] == pytest.approx(0.0)


def test_output_data_empty_matrices_dict() -> None:
    """Verify empty matrices dict is handled correctly."""
    output: OutputData = OutputData()

    assert output.matrices == {}

    data: Dict[str, object] = output.to_dict(include_matrices=True)
    assert "matrices" not in data


def test_output_data_empty_extra_dict() -> None:
    """Verify empty extra dict is handled correctly."""
    output: OutputData = OutputData()

    assert output.extra == {}

    data: Dict[str, object] = output.to_dict()
    assert "extra" not in data
