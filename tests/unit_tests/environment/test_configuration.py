"""Unit tests for q_block.configuration module.

Tests cover:
- PathConfiguration: directory paths for inputs and outputs
- CalculationDefaults: default SCF parameters
- OutputSettings: output format configuration
- Configuration: main configuration container
- Configuration.from_file: loading from JSON
- Configuration.from_env: loading from environment variables
- Configuration.save_to_file: saving to JSON

All tests use pytest with parametrize, no test classes.
"""

import json
import os
from pathlib import Path
from typing import Dict

import pytest

from q_block.environment.configuration import (
    CalculationDefaults,
    Configuration,
    OutputSettings,
    PathConfiguration,
)


# ======================================================================
# PathConfiguration Tests
# ======================================================================


def test_path_configuration_defaults() -> None:
    """Verify PathConfiguration creates default paths correctly."""
    paths: PathConfiguration = PathConfiguration()

    assert isinstance(paths.basis_set_dir, Path)
    assert isinstance(paths.geometry_dir, Path)
    assert isinstance(paths.output_dir, Path)
    assert isinstance(paths.log_dir, Path)
    assert "basis_set" in str(paths.basis_set_dir)


def test_path_configuration_custom_paths() -> None:
    """Verify PathConfiguration accepts custom path specifications."""
    paths: PathConfiguration = PathConfiguration(
        basis_set_dir=Path("custom/basis"),
        geometry_dir=Path("custom/geom"),
        output_dir=Path("custom/out"),
        log_dir=Path("custom/logs"),
    )

    assert paths.basis_set_dir == Path("custom/basis")
    assert paths.geometry_dir == Path("custom/geom")
    assert paths.output_dir == Path("custom/out")
    assert paths.log_dir == Path("custom/logs")


def test_path_configuration_ensure_directories(tmp_path: Path) -> None:
    """Verify ensure_directories creates missing directories.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    paths: PathConfiguration = PathConfiguration(
        output_dir=tmp_path / "output",
        log_dir=tmp_path / "logs",
    )

    assert not paths.output_dir.exists()
    assert not paths.log_dir.exists()

    paths.ensure_directories()

    assert paths.output_dir.exists()
    assert paths.log_dir.exists()
    assert paths.output_dir.is_dir()
    assert paths.log_dir.is_dir()


# ======================================================================
# CalculationDefaults Tests
# ======================================================================


def test_calculation_defaults_values() -> None:
    """Verify CalculationDefaults has sensible default values."""
    calc: CalculationDefaults = CalculationDefaults()

    assert calc.max_scf_iterations == 100
    assert calc.scf_convergence_threshold == pytest.approx(1e-8)
    assert calc.diis_start == 1
    assert calc.diis_max_vectors == 6
    assert calc.error_metric == "rms"


def test_calculation_defaults_custom_values() -> None:
    """Verify CalculationDefaults accepts custom values."""
    calc: CalculationDefaults = CalculationDefaults(
        max_scf_iterations=200,
        scf_convergence_threshold=1e-10,
        diis_start=3,
        diis_max_vectors=10,
        error_metric="max",
    )

    assert calc.max_scf_iterations == 200
    assert calc.scf_convergence_threshold == pytest.approx(1e-10)
    assert calc.diis_start == 3
    assert calc.diis_max_vectors == 10
    assert calc.error_metric == "max"


# ======================================================================
# OutputSettings Tests
# ======================================================================


def test_output_settings_defaults() -> None:
    """Verify OutputSettings has correct default values."""
    settings: OutputSettings = OutputSettings()

    assert settings.save_json is True
    assert settings.save_text is True
    assert settings.save_log is True
    assert settings.save_matrices is False
    assert settings.log_level == "INFO"
    assert settings.json_indent == 2
    assert settings.output_prefix == "output"


def test_output_settings_custom() -> None:
    """Verify OutputSettings accepts custom values."""
    settings: OutputSettings = OutputSettings(
        save_json=False,
        save_text=False,
        save_log=False,
        save_matrices=True,
        log_level="DEBUG",
        json_indent=4,
        output_prefix="custom_output",
    )

    assert settings.save_json is False
    assert settings.save_text is False
    assert settings.save_log is False
    assert settings.save_matrices is True
    assert settings.log_level == "DEBUG"
    assert settings.json_indent == 4
    assert settings.output_prefix == "custom_output"


# ======================================================================
# Configuration Tests
# ======================================================================


def test_configuration_init_defaults() -> None:
    """Verify Configuration initializes with all default sub-configs."""
    config: Configuration = Configuration()

    assert isinstance(config.paths, PathConfiguration)
    assert isinstance(config.calculation, CalculationDefaults)
    assert isinstance(config.output, OutputSettings)


def test_configuration_convenience_overrides() -> None:
    """Verify Configuration convenience parameters override defaults."""
    config: Configuration = Configuration(
        output_dir="custom_output",
        max_scf_iterations=150,
        scf_convergence_threshold=1e-9,
    )

    assert config.paths.output_dir == Path("custom_output")
    assert config.calculation.max_scf_iterations == 150
    assert config.calculation.scf_convergence_threshold == pytest.approx(1e-9)


def test_configuration_repr() -> None:
    """Verify Configuration.__repr__ shows key settings."""
    config: Configuration = Configuration(
        output_dir="test_out",
        max_scf_iterations=75,
    )

    repr_str: str = repr(config)

    assert "Configuration(" in repr_str
    assert "output_dir=test_out" in repr_str
    assert "max_scf_iterations=75" in repr_str


# ======================================================================
# Configuration.from_file Tests
# ======================================================================


def test_configuration_from_file_valid(tmp_path: Path) -> None:
    """Verify from_file loads configuration from JSON file.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_data: Dict = {
        "paths": {
            "output_dir": "file_output",
            "log_dir": "file_logs",
        },
        "calculation": {
            "max_scf_iterations": 250,
            "scf_convergence_threshold": 1e-10,
        },
        "output": {
            "log_level": "DEBUG",
            "save_matrices": True,
        },
    }

    config_file: Path = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.paths.output_dir == Path("file_output")
    assert config.paths.log_dir == Path("file_logs")
    assert config.calculation.max_scf_iterations == 250
    assert config.calculation.scf_convergence_threshold == pytest.approx(1e-10)
    assert config.output.log_level == "DEBUG"
    assert config.output.save_matrices is True


def test_configuration_from_file_not_found() -> None:
    """Verify from_file raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        Configuration.from_file("nonexistent_config.json")


def test_configuration_from_file_invalid_json(tmp_path: Path) -> None:
    """Verify from_file raises ValueError for malformed JSON.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    bad_file: Path = tmp_path / "bad.json"
    bad_file.write_text("{ invalid json }", encoding="utf-8")

    with pytest.raises(ValueError):
        Configuration.from_file(bad_file)


def test_configuration_from_file_partial(tmp_path: Path) -> None:
    """Verify from_file handles partial configuration (uses defaults).

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config_data: Dict = {
        "calculation": {
            "max_scf_iterations": 50,
        }
    }

    config_file: Path = tmp_path / "partial.json"
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    config: Configuration = Configuration.from_file(config_file)

    assert config.calculation.max_scf_iterations == 50
    assert config.calculation.scf_convergence_threshold == pytest.approx(1e-8)
    assert config.output.save_json is True


# ======================================================================
# Configuration.from_env Tests
# ======================================================================


def test_configuration_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify from_env uses defaults when no env vars set.

    :param monkeypatch: Pytest fixture for modifying environment.
    :type monkeypatch: pytest.MonkeyPatch
    """
    for key in list(os.environ.keys()):
        if key.startswith("QBLOCK_"):
            monkeypatch.delenv(key, raising=False)

    config: Configuration = Configuration.from_env()

    assert config.calculation.max_scf_iterations == 100
    assert config.output.log_level == "INFO"


def test_configuration_from_env_with_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify from_env reads configuration from environment variables.

    :param monkeypatch: Pytest fixture for modifying environment.
    :type monkeypatch: pytest.MonkeyPatch
    """
    monkeypatch.setenv("QBLOCK_OUTPUT_DIR", "env_output")
    monkeypatch.setenv("QBLOCK_MAX_SCF_ITERATIONS", "175")
    monkeypatch.setenv("QBLOCK_SCF_CONVERGENCE_THRESHOLD", "1e-9")
    monkeypatch.setenv("QBLOCK_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("QBLOCK_SAVE_MATRICES", "true")
    monkeypatch.setenv("QBLOCK_OUTPUT_PREFIX", "test_prefix")

    config: Configuration = Configuration.from_env()

    assert config.paths.output_dir == Path("env_output")
    assert config.calculation.max_scf_iterations == 175
    assert config.calculation.scf_convergence_threshold == pytest.approx(1e-9)
    assert config.output.log_level == "DEBUG"
    assert config.output.save_matrices is True
    assert config.output.output_prefix == "test_prefix"


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("", False),
        ("anything_else", False),
    ],
    ids=["true", "True", "TRUE", "false", "False", "FALSE", "empty", "other"],
)
def test_configuration_from_env_bool_parsing(
    monkeypatch: pytest.MonkeyPatch, env_value: str, expected: bool
) -> None:
    """Verify from_env correctly parses boolean environment variables.

    :param monkeypatch: Pytest fixture for modifying environment.
    :type monkeypatch: pytest.MonkeyPatch
    :param env_value: String value in environment variable.
    :type env_value: str
    :param expected: Expected boolean result.
    :type expected: bool
    """
    monkeypatch.setenv("QBLOCK_SAVE_JSON", env_value)

    config: Configuration = Configuration.from_env()

    assert config.output.save_json == expected


# ======================================================================
# Configuration.save_to_file Tests
# ======================================================================


def test_configuration_save_to_file(tmp_path: Path) -> None:
    """Verify save_to_file creates valid JSON configuration file.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    config: Configuration = Configuration(
        output_dir="saved_output",
        max_scf_iterations=125,
        scf_convergence_threshold=5e-9,
    )

    save_path: Path = tmp_path / "saved_config.json"
    config.save_to_file(save_path)

    assert save_path.exists()

    loaded_config: Configuration = Configuration.from_file(save_path)

    assert loaded_config.paths.output_dir == Path("saved_output")
    assert loaded_config.calculation.max_scf_iterations == 125
    assert loaded_config.calculation.scf_convergence_threshold == pytest.approx(5e-9)


def test_configuration_roundtrip(tmp_path: Path) -> None:
    """Verify configuration can be saved and loaded without data loss.

    :param tmp_path: Pytest fixture providing a temporary directory.
    :type tmp_path: Path
    """
    original: Configuration = Configuration(
        output_dir="roundtrip_output",
        max_scf_iterations=333,
        scf_convergence_threshold=7.5e-10,
    )
    original.output.log_level = "WARNING"
    original.output.output_prefix = "roundtrip_test"

    save_path: Path = tmp_path / "roundtrip.json"
    original.save_to_file(save_path)

    loaded: Configuration = Configuration.from_file(save_path)

    assert loaded.paths.output_dir == original.paths.output_dir
    assert loaded.calculation.max_scf_iterations == original.calculation.max_scf_iterations
    assert loaded.calculation.scf_convergence_threshold == pytest.approx(
        original.calculation.scf_convergence_threshold
    )
    assert loaded.output.log_level == original.output.log_level
    assert loaded.output.output_prefix == original.output.output_prefix
