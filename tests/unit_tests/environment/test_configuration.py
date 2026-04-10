"""Unit tests for q_block.environment.configuration module.

Tests cover:
- Configuration: default values and programmatic creation
- Configuration.from_file: loading from plain-text config files
- Configuration._parse: inline geometry, geometry_file, key=value parsing
- _cast_value: type casting for recognised keys
- _parse_geometry_line: per-atom parsing with global fallbacks
- Error handling: unknown keys, malformed geometry, missing $end

All tests use pytest with parametrize, no test classes.
"""

from pathlib import Path
from typing import Any, List

import pytest

from q_block.environment.configuration import (
    Configuration,
    _cast_value,
    _parse_geometry_line,
)


# ======================================================================
# Default values
# ======================================================================


def test_configuration_defaults() -> None:
    """Verify Configuration initializes with correct defaults."""
    config: Configuration = Configuration()

    assert config.basis_set == "STO-3G"
    assert config.charge == 0
    assert config.multiplicity == 1
    assert config.geometry_file is None
    assert config.geometry == []
    assert config.max_scf_iterations == 100
    assert config.scf_convergence_threshold == pytest.approx(1e-8)
    assert config.diis_start == 1
    assert config.diis_max_vectors == 6
    assert config.error_metric == "rms"
    assert config.output_dir == "output"
    assert config.log_level == "INFO"
    assert config.save_json is True
    assert config.save_text is True


def test_configuration_custom_values() -> None:
    """Verify Configuration accepts custom keyword arguments."""
    config: Configuration = Configuration(
        basis_set="3-21G",
        charge=1,
        multiplicity=2,
        max_scf_iterations=200,
        scf_convergence_threshold=1e-10,
        diis_start=3,
        diis_max_vectors=10,
        error_metric="max",
        output_dir="custom_output",
        log_level="DEBUG",
        save_json=False,
        save_text=False,
    )

    assert config.basis_set == "3-21G"
    assert config.charge == 1
    assert config.multiplicity == 2
    assert config.max_scf_iterations == 200
    assert config.scf_convergence_threshold == pytest.approx(1e-10)
    assert config.diis_start == 3
    assert config.diis_max_vectors == 10
    assert config.error_metric == "max"
    assert config.output_dir == "custom_output"
    assert config.log_level == "DEBUG"
    assert config.save_json is False
    assert config.save_text is False


def test_configuration_geometry_list() -> None:
    """Verify Configuration stores inline geometry list."""
    geom: List[List[Any]] = [
        ["H", 0.0, 0.0, 0.0],
        ["H", 0.0, 0.0, 0.74],
    ]
    config: Configuration = Configuration(geometry=geom)

    assert len(config.geometry) == 2
    assert config.geometry[0][0] == "H"
    assert config.geometry[1][3] == pytest.approx(0.74)


def test_configuration_repr() -> None:
    """Verify Configuration.__repr__ shows key settings."""
    config: Configuration = Configuration(
        basis_set="3-21G",
        charge=1,
        multiplicity=2,
        max_scf_iterations=75,
    )

    repr_str: str = repr(config)

    assert "Configuration(" in repr_str
    assert "3-21G" in repr_str
    assert "max_scf_iterations=75" in repr_str


# ======================================================================
# _cast_value
# ======================================================================


@pytest.mark.parametrize(
    "key, raw, expected",
    [
        ("basis_set", "3-21G", "3-21G"),
        ("charge", "2", 2),
        ("multiplicity", "3", 3),
        ("max_scf_iterations", "200", 200),
        ("scf_convergence_threshold", "1e-10", 1e-10),
        ("diis_start", "5", 5),
        ("diis_max_vectors", "8", 8),
        ("error_metric", "max", "max"),
        ("output_dir", "results", "results"),
        ("log_level", "DEBUG", "DEBUG"),
        ("save_json", "true", True),
        ("save_json", "false", False),
        ("save_json", "True", True),
        ("save_text", "yes", True),
        ("save_text", "0", False),
    ],
    ids=[
        "str_basis_set",
        "int_charge",
        "int_multiplicity",
        "int_max_iter",
        "float_threshold",
        "int_diis_start",
        "int_diis_max",
        "str_error_metric",
        "str_output_dir",
        "str_log_level",
        "bool_true",
        "bool_false",
        "bool_True",
        "bool_yes",
        "bool_zero",
    ],
)
def test_cast_value(
    key: str, raw: str, expected: Any
) -> None:
    """Verify _cast_value converts strings to correct types.

    :param key: Configuration key name.
    :type key: str
    :param raw: Raw string value.
    :type raw: str
    :param expected: Expected typed value.
    :type expected: Any
    """
    result: Any = _cast_value(key, raw)

    if isinstance(expected, float):
        assert result == pytest.approx(expected)
    else:
        assert result == expected


def test_cast_value_unknown_key() -> None:
    """Verify _cast_value raises ValueError for unknown keys."""
    with pytest.raises(ValueError, match="Unknown configuration key"):
        _cast_value("nonexistent_key", "value")


# ======================================================================
# _parse_geometry_line
# ======================================================================


def test_parse_geometry_line_minimal() -> None:
    """Verify parsing a 4-field geometry line with global basis."""
    result: List[Any] = _parse_geometry_line(
        "H  0.0  0.0  0.74", "STO-3G"
    )

    assert result[0] == "H"
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)
    assert result[3] == pytest.approx(0.74)
    assert result[4] == "STO-3G"


def test_parse_geometry_line_per_atom_basis() -> None:
    """Verify per-atom basis set overrides global."""
    result: List[Any] = _parse_geometry_line(
        "O  0.0  0.0  0.0  6-31G", "STO-3G"
    )

    assert result[0] == "O"
    assert result[4] == "6-31G"


def test_parse_geometry_line_per_atom_charge() -> None:
    """Verify per-atom charge is parsed."""
    result: List[Any] = _parse_geometry_line(
        "Fe  0.0  0.0  0.0  STO-3G  2", "STO-3G"
    )

    assert result[0] == "Fe"
    assert result[4] == "STO-3G"
    assert result[5] == 2


def test_parse_geometry_line_no_global_basis() -> None:
    """Verify 4-field line without global basis returns 4 items."""
    result: List[Any] = _parse_geometry_line(
        "H  0.0  0.0  0.0", None
    )

    assert len(result) == 4
    assert result[0] == "H"


def test_parse_geometry_line_too_few_fields() -> None:
    """Verify ValueError for fewer than 4 fields."""
    with pytest.raises(ValueError, match="at least 4 fields"):
        _parse_geometry_line("H  0.0  0.0", "STO-3G")


def test_parse_geometry_line_bad_coordinates() -> None:
    """Verify ValueError for non-numeric coordinates."""
    with pytest.raises(ValueError, match="Coordinates must be numeric"):
        _parse_geometry_line("H  abc  0.0  0.0", "STO-3G")


def test_parse_geometry_line_bad_charge() -> None:
    """Verify ValueError for non-integer per-atom charge."""
    with pytest.raises(ValueError, match="Per-atom charge"):
        _parse_geometry_line(
            "H  0.0  0.0  0.0  STO-3G  abc", "STO-3G"
        )


# ======================================================================
# Configuration.from_file
# ======================================================================


def test_from_file_full(tmp_path: Path) -> None:
    """Verify from_file loads all settings from a config file.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "basis_set = 3-21G\n"
        "charge = 1\n"
        "multiplicity = 2\n"
        "max_scf_iterations = 250\n"
        "scf_convergence_threshold = 1e-10\n"
        "diis_start = 3\n"
        "diis_max_vectors = 10\n"
        "error_metric = max\n"
        "output_dir = results\n"
        "log_level = DEBUG\n"
        "save_json = false\n"
        "save_text = false\n"
        "\n"
        "$geometry\n"
        "H  0.0  0.0  0.0\n"
        "H  0.0  0.0  0.74\n"
        "$end\n"
    )

    config_file: Path = tmp_path / "test.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.basis_set == "3-21G"
    assert config.charge == 1
    assert config.multiplicity == 2
    assert config.max_scf_iterations == 250
    assert config.scf_convergence_threshold == pytest.approx(1e-10)
    assert config.diis_start == 3
    assert config.diis_max_vectors == 10
    assert config.error_metric == "max"
    assert config.output_dir == "results"
    assert config.log_level == "DEBUG"
    assert config.save_json is False
    assert config.save_text is False
    assert len(config.geometry) == 2
    assert config.geometry[0][0] == "H"
    assert config.geometry[1][3] == pytest.approx(0.74)


def test_from_file_partial(tmp_path: Path) -> None:
    """Verify from_file uses defaults for unspecified keys.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = "max_scf_iterations = 50\n"

    config_file: Path = tmp_path / "partial.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.max_scf_iterations == 50
    assert config.basis_set == "STO-3G"
    assert config.scf_convergence_threshold == pytest.approx(1e-8)
    assert config.save_json is True


def test_from_file_not_found() -> None:
    """Verify from_file raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        Configuration.from_file("nonexistent.qblock.config")


def test_from_file_no_default_returns_defaults() -> None:
    """Verify from_file with no argument returns defaults when .qblock.config missing."""
    config: Configuration = Configuration.from_file()

    assert config.basis_set == "STO-3G"
    assert config.max_scf_iterations == 100


def test_from_file_comments_and_blanks(tmp_path: Path) -> None:
    """Verify comments and blank lines are ignored.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "# This is a comment\n"
        "\n"
        "basis_set = 3-21G\n"
        "# Another comment\n"
        "\n"
        "charge = 0\n"
    )

    config_file: Path = tmp_path / "comments.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.basis_set == "3-21G"
    assert config.charge == 0


def test_from_file_geometry_file_key(tmp_path: Path) -> None:
    """Verify geometry_file key is parsed correctly.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = "geometry_file = geometries/water.xyz\n"

    config_file: Path = tmp_path / "geom_file.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.geometry_file == "geometries/water.xyz"
    assert config.geometry == []


def test_from_file_per_atom_overrides(tmp_path: Path) -> None:
    """Verify per-atom basis_set and charge override globals.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "basis_set = STO-3G\n"
        "\n"
        "$geometry\n"
        "H  0.0  0.0  0.0\n"
        "O  0.0  0.0  1.0  6-31G  -1\n"
        "$end\n"
    )

    config_file: Path = tmp_path / "overrides.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert len(config.geometry) == 2
    # H atom gets global basis_set
    assert config.geometry[0][4] == "STO-3G"
    # O atom has per-atom overrides
    assert config.geometry[1][4] == "6-31G"
    assert config.geometry[1][5] == -1


# ======================================================================
# Error handling
# ======================================================================


def test_from_file_unknown_key(tmp_path: Path) -> None:
    """Verify ValueError for unknown configuration keys.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = "unknown_key = value\n"

    config_file: Path = tmp_path / "bad_key.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown configuration key"):
        Configuration.from_file(config_file)


def test_from_file_unclosed_geometry(tmp_path: Path) -> None:
    """Verify ValueError for unclosed $geometry block.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "$geometry\n"
        "H  0.0  0.0  0.0\n"
    )

    config_file: Path = tmp_path / "unclosed.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Unclosed"):
        Configuration.from_file(config_file)


def test_from_file_nested_geometry(tmp_path: Path) -> None:
    """Verify ValueError for nested $geometry blocks.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "$geometry\n"
        "$geometry\n"
        "$end\n"
        "$end\n"
    )

    config_file: Path = tmp_path / "nested.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="nested"):
        Configuration.from_file(config_file)


def test_from_file_end_without_geometry(tmp_path: Path) -> None:
    """Verify ValueError for $end without $geometry.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = "$end\n"

    config_file: Path = tmp_path / "orphan_end.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="without"):
        Configuration.from_file(config_file)


def test_from_file_missing_equals(tmp_path: Path) -> None:
    """Verify ValueError for lines without '=' outside geometry.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = "this is not a valid line\n"

    config_file: Path = tmp_path / "bad_syntax.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="key = value"):
        Configuration.from_file(config_file)


# ======================================================================
# Getter methods
# ======================================================================


def test_get_basis_set_returns_str() -> None:
    """Verify get_basis_set returns str."""
    config: Configuration = Configuration(basis_set="6-31G")

    result: str = config.get_basis_set()

    assert result == "6-31G"
    assert isinstance(result, str)


def test_get_charge_returns_int() -> None:
    """Verify get_charge returns int."""
    config: Configuration = Configuration(charge=-1)

    assert config.get_charge() == -1
    assert isinstance(config.get_charge(), int)


def test_get_multiplicity_returns_int() -> None:
    """Verify get_multiplicity returns int."""
    config: Configuration = Configuration(multiplicity=3)

    assert config.get_multiplicity() == 3
    assert isinstance(config.get_multiplicity(), int)


def test_get_geometry_file_returns_path() -> None:
    """Verify get_geometry_file returns Path when set."""
    config: Configuration = Configuration(
        geometry_file="geometries/water.xyz"
    )

    result: Optional[Path] = config.get_geometry_file()

    assert result is not None
    assert isinstance(result, Path)
    assert result == Path("geometries/water.xyz")


def test_get_geometry_file_returns_none() -> None:
    """Verify get_geometry_file returns None when unset."""
    config: Configuration = Configuration()

    assert config.get_geometry_file() is None


def test_get_geometry_returns_list() -> None:
    """Verify get_geometry returns geometry list."""
    geom: List[List[Any]] = [
        ["H", 0.0, 0.0, 0.0],
        ["H", 0.0, 0.0, 0.74],
    ]
    config: Configuration = Configuration(geometry=geom)

    result: List[List[Any]] = config.get_geometry()

    assert len(result) == 2
    assert result[0][0] == "H"
    assert result[1][3] == pytest.approx(0.74)


def test_get_geometry_empty_default() -> None:
    """Verify get_geometry returns empty list by default."""
    config: Configuration = Configuration()

    assert config.get_geometry() == []


def test_get_max_scf_iterations_returns_int() -> None:
    """Verify get_max_scf_iterations returns int."""
    config: Configuration = Configuration(
        max_scf_iterations=200
    )

    assert config.get_max_scf_iterations() == 200
    assert isinstance(config.get_max_scf_iterations(), int)


def test_get_scf_convergence_threshold_returns_float() -> None:
    """Verify get_scf_convergence_threshold returns float."""
    config: Configuration = Configuration(
        scf_convergence_threshold=1e-10
    )

    assert config.get_scf_convergence_threshold() == pytest.approx(
        1e-10
    )
    assert isinstance(
        config.get_scf_convergence_threshold(), float
    )


def test_get_diis_start_returns_int() -> None:
    """Verify get_diis_start returns int."""
    config: Configuration = Configuration(diis_start=5)

    assert config.get_diis_start() == 5
    assert isinstance(config.get_diis_start(), int)


def test_get_diis_max_vectors_returns_int() -> None:
    """Verify get_diis_max_vectors returns int."""
    config: Configuration = Configuration(diis_max_vectors=10)

    assert config.get_diis_max_vectors() == 10
    assert isinstance(config.get_diis_max_vectors(), int)


def test_get_error_metric_returns_str() -> None:
    """Verify get_error_metric returns str."""
    config: Configuration = Configuration(error_metric="max")

    assert config.get_error_metric() == "max"
    assert isinstance(config.get_error_metric(), str)


def test_get_output_dir_returns_path() -> None:
    """Verify get_output_dir returns Path."""
    config: Configuration = Configuration(
        output_dir="custom_results"
    )

    result: Path = config.get_output_dir()

    assert isinstance(result, Path)
    assert result == Path("custom_results")


def test_get_output_dir_default_returns_path() -> None:
    """Verify get_output_dir returns Path for default."""
    config: Configuration = Configuration()

    result: Path = config.get_output_dir()

    assert isinstance(result, Path)
    assert result == Path("output")


def test_get_log_level_returns_str() -> None:
    """Verify get_log_level returns str."""
    config: Configuration = Configuration(log_level="DEBUG")

    assert config.get_log_level() == "DEBUG"
    assert isinstance(config.get_log_level(), str)


def test_get_save_json_returns_bool() -> None:
    """Verify get_save_json returns bool."""
    config: Configuration = Configuration(save_json=False)

    assert config.get_save_json() is False
    assert isinstance(config.get_save_json(), bool)


def test_get_save_text_returns_bool() -> None:
    """Verify get_save_text returns bool."""
    config: Configuration = Configuration(save_text=False)

    assert config.get_save_text() is False
    assert isinstance(config.get_save_text(), bool)


def test_getters_after_from_file(tmp_path: Path) -> None:
    """Verify getter methods work on file-loaded config.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_text: str = (
        "basis_set = 3-21G\n"
        "charge = 1\n"
        "multiplicity = 2\n"
        "output_dir = results\n"
        "geometry_file = mol.xyz\n"
        "save_json = false\n"
    )

    config_file: Path = tmp_path / "getter.qblock.config"
    config_file.write_text(config_text, encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.get_basis_set() == "3-21G"
    assert config.get_charge() == 1
    assert config.get_multiplicity() == 2
    assert config.get_output_dir() == Path("results")
    assert isinstance(config.get_output_dir(), Path)
    assert config.get_geometry_file() == Path("mol.xyz")
    assert isinstance(config.get_geometry_file(), Path)
    assert config.get_save_json() is False
    assert config.get_save_text() is True
