"""Configuration management for qBlock calculations.

This module provides centralized configuration for:
- File paths (input, output, basis sets, geometries)
- Calculation defaults (convergence thresholds, max iterations)
- Output settings (log levels, file formats)

Design Philosophy
-----------------
- **Environment-aware**: reads from environment variables and config files
- **Type-safe**: all configuration values are typed and validated
- **Layered defaults**: system defaults → config file → environment variables
- **No global state**: configuration is passed explicitly, not imported globally

Usage
-----
1. **Default configuration** (recommended for most cases)::

    from q_block.environment import Configuration

    config = Configuration()
    # Uses default paths and settings

2. **Load from default .config file**::

    config = Configuration.from_file()
    # Searches for .qblock.config, qblock.config.json, or .config

3. **Custom configuration from file**::

    config = Configuration.from_file("my_config.json")

4. **Override specific settings**::

    config = Configuration(
        output_dir="custom_output",
        max_scf_iterations=200,
    )

5. **From environment variables**::

    # Set environment variables:
    # QBLOCK_OUTPUT_DIR=results
    # QBLOCK_MAX_SCF_ITERATIONS=150

    config = Configuration.from_env()
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


@dataclass
class PathConfiguration:
    """File path configuration for inputs and outputs.

    :param basis_set_dir: Directory containing basis set files (.gbs).
    :type basis_set_dir: Path
    :param geometry_dir: Directory for molecular geometry files (.xyz).
    :type geometry_dir: Path
    :param output_dir: Directory for calculation outputs.
    :type output_dir: Path
    :param log_dir: Directory for log files.
    :type log_dir: Path
    """

    basis_set_dir: Path = field(
        default_factory=lambda: Path("q_block/environment/constants/numerical/basis_set")
    )
    geometry_dir: Path = field(default_factory=lambda: Path("geometries"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))

    def ensure_directories(self) -> None:
        """Create output and log directories if they don't exist.

        :returns: None
        :rtype: None
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CalculationDefaults:
    """Default parameters for quantum chemistry calculations.

    :param max_scf_iterations: Maximum SCF iterations before stop.
    :type max_scf_iterations: int
    :param scf_convergence_threshold: Energy and error convergence threshold.
    :type scf_convergence_threshold: float
    :param diis_start: Iteration to begin DIIS extrapolation (0-indexed).
    :type diis_start: int
    :param diis_max_vectors: Maximum DIIS subspace size.
    :type diis_max_vectors: int
    :param error_metric: Error metric for convergence ("rms" or "max").
    :type error_metric: str
    """

    max_scf_iterations: int = 100
    scf_convergence_threshold: float = 1e-8
    diis_start: int = 1
    diis_max_vectors: int = 6
    error_metric: str = "rms"


@dataclass
class OutputSettings:
    """Output format and logging configuration.

    :param save_json: Whether to save JSON output by default.
    :type save_json: bool
    :param save_text: Whether to save text summary by default.
    :type save_text: bool
    :param save_log: Whether to save complete log file.
    :type save_log: bool
    :param save_matrices: Whether to include matrices in JSON output.
    :type save_matrices: bool
    :param log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    :type log_level: str
    :param json_indent: Indentation spaces for JSON output.
    :type json_indent: int
    :param output_prefix: Prefix for all output filenames.
    :type output_prefix: str
    """

    save_json: bool = True
    save_text: bool = True
    save_log: bool = True
    save_matrices: bool = False
    log_level: str = "INFO"
    json_indent: int = 2
    output_prefix: str = "output"


class Configuration:
    """Main configuration container for qBlock calculations.

    Aggregates all configuration sections and provides methods for
    loading from files and environment variables.

    :param paths: Path configuration.
    :type paths: Optional[PathConfiguration]
    :param calculation: Calculation default parameters.
    :type calculation: Optional[CalculationDefaults]
    :param output: Output settings.
    :type output: Optional[OutputSettings]
    """

    def __init__(
        self,
        paths: Optional[PathConfiguration] = None,
        calculation: Optional[CalculationDefaults] = None,
        output: Optional[OutputSettings] = None,
        # Convenience parameters for path overrides
        basis_set_dir: Optional[Union[str, Path]] = None,
        geometry_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        # Convenience parameters for calculation overrides
        max_scf_iterations: Optional[int] = None,
        scf_convergence_threshold: Optional[float] = None,
    ) -> None:
        # Initialize with defaults or provided configurations
        self.paths: PathConfiguration = paths or PathConfiguration()
        self.calculation: CalculationDefaults = (
            calculation or CalculationDefaults()
        )
        self.output: OutputSettings = output or OutputSettings()

        # Apply convenience overrides
        if basis_set_dir is not None:
            self.paths.basis_set_dir = Path(basis_set_dir)
        if geometry_dir is not None:
            self.paths.geometry_dir = Path(geometry_dir)
        if output_dir is not None:
            self.paths.output_dir = Path(output_dir)
        if log_dir is not None:
            self.paths.log_dir = Path(log_dir)
        if max_scf_iterations is not None:
            self.calculation.max_scf_iterations = max_scf_iterations
        if scf_convergence_threshold is not None:
            self.calculation.scf_convergence_threshold = scf_convergence_threshold

    @classmethod
    def from_file(cls, filepath: Optional[Union[str, Path]] = None) -> "Configuration":
        """Load configuration from a JSON file.

        If no filepath is provided, attempts to load from default locations:
        1. .qblock.config in current directory
        2. qblock.config.json in current directory
        3. .config in current directory (fallback)

        File format::

            {
              "paths": {
                "basis_set_dir": "path/to/basis",
                "output_dir": "path/to/output"
              },
              "calculation": {
                "max_scf_iterations": 150,
                "scf_convergence_threshold": 1e-9
              },
              "output": {
                "save_json": true,
                "log_level": "DEBUG"

              }
            }

        :param filepath: Path to JSON configuration file. If None, uses default locations.
        :type filepath: Optional[Union[str, Path]]
        :returns: Configuration instance loaded from file.
        :rtype: Configuration
        :raises FileNotFoundError: If configuration file does not exist.
        :raises ValueError: If JSON is malformed.
        """
        # If no filepath provided, try default locations
        if filepath is None:
            default_paths = [
                Path(".qblock.config"),
                Path("qblock.config.json"),
                Path(".config"),
            ]
            path = None
            for default_path in default_paths:
                if default_path.exists():
                    path = default_path
                    break
            
            if path is None:
                # No default config found, return configuration with defaults
                return cls()
        else:
            path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        # Build configuration sections
        paths = None
        if "paths" in data:
            paths = PathConfiguration(
                basis_set_dir=Path(data["paths"].get("basis_set_dir", "q_block/environment/constants/numerical/basis_set")),
                geometry_dir=Path(data["paths"].get("geometry_dir", "geometries")),
                output_dir=Path(data["paths"].get("output_dir", "output")),
                log_dir=Path(data["paths"].get("log_dir", "logs")),
            )

        calculation = None
        if "calculation" in data:
            calculation = CalculationDefaults(
                max_scf_iterations=data["calculation"].get("max_scf_iterations", 100),
                scf_convergence_threshold=data["calculation"].get("scf_convergence_threshold", 1e-8),
                diis_start=data["calculation"].get("diis_start", 1),
                diis_max_vectors=data["calculation"].get("diis_max_vectors", 6),
                error_metric=data["calculation"].get("error_metric", "rms"),
            )

        output_cfg = None
        if "output" in data:
            output_cfg = OutputSettings(
                save_json=data["output"].get("save_json", True),
                save_text=data["output"].get("save_text", True),
                save_log=data["output"].get("save_log", True),
                save_matrices=data["output"].get("save_matrices", False),
                log_level=data["output"].get("log_level", "INFO"),
                json_indent=data["output"].get("json_indent", 2),
                output_prefix=data["output"].get("output_prefix", "output"),
            )

        return cls(
            paths=paths,
            calculation=calculation,
            output=output_cfg,
        )

    @classmethod
    def from_env(cls) -> "Configuration":
        """Load configuration from environment variables.

        Supported environment variables::

            QBLOCK_BASIS_SET_DIR
            QBLOCK_GEOMETRY_DIR
            QBLOCK_OUTPUT_DIR
            QBLOCK_LOG_DIR
            QBLOCK_MAX_SCF_ITERATIONS
            QBLOCK_SCF_CONVERGENCE_THRESHOLD
            QBLOCK_DIIS_START
            QBLOCK_DIIS_MAX_VECTORS
            QBLOCK_ERROR_METRIC
            QBLOCK_LOG_LEVEL
            QBLOCK_SAVE_JSON
            QBLOCK_SAVE_TEXT
            QBLOCK_SAVE_LOG
            QBLOCK_SAVE_MATRICES
            QBLOCK_OUTPUT_PREFIX

        :returns: Configuration instance from environment variables.
        :rtype: Configuration
        """
        paths = PathConfiguration(
            basis_set_dir=Path(os.getenv("QBLOCK_BASIS_SET_DIR", "q_block/environment/constants/numerical/basis_set")),
            geometry_dir=Path(os.getenv("QBLOCK_GEOMETRY_DIR", "geometries")),
            output_dir=Path(os.getenv("QBLOCK_OUTPUT_DIR", "output")),
            log_dir=Path(os.getenv("QBLOCK_LOG_DIR", "logs")),
        )

        calculation = CalculationDefaults(
            max_scf_iterations=int(os.getenv("QBLOCK_MAX_SCF_ITERATIONS", "100")),
            scf_convergence_threshold=float(os.getenv("QBLOCK_SCF_CONVERGENCE_THRESHOLD", "1e-8")),
            diis_start=int(os.getenv("QBLOCK_DIIS_START", "1")),
            diis_max_vectors=int(os.getenv("QBLOCK_DIIS_MAX_VECTORS", "6")),
            error_metric=os.getenv("QBLOCK_ERROR_METRIC", "rms"),
        )

        output_cfg = OutputSettings(
            save_json=os.getenv("QBLOCK_SAVE_JSON", "true").lower() == "true",
            save_text=os.getenv("QBLOCK_SAVE_TEXT", "true").lower() == "true",
            save_log=os.getenv("QBLOCK_SAVE_LOG", "true").lower() == "true",
            save_matrices=os.getenv("QBLOCK_SAVE_MATRICES", "false").lower() == "true",
            log_level=os.getenv("QBLOCK_LOG_LEVEL", "INFO"),
            json_indent=int(os.getenv("QBLOCK_JSON_INDENT", "2")),
            output_prefix=os.getenv("QBLOCK_OUTPUT_PREFIX", "output"),
        )

        return cls(
            paths=paths,
            calculation=calculation,
            output=output_cfg,
        )

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current configuration to a JSON file.

        :param filepath: Path where to save configuration.
        :type filepath: Union[str, Path]
        :returns: None
        :rtype: None
        """
        data = {
            "paths": {
                "basis_set_dir": str(self.paths.basis_set_dir),
                "geometry_dir": str(self.paths.geometry_dir),
                "output_dir": str(self.paths.output_dir),
                "log_dir": str(self.paths.log_dir),
            },
            "calculation": {
                "max_scf_iterations": self.calculation.max_scf_iterations,
                "scf_convergence_threshold": self.calculation.scf_convergence_threshold,
                "diis_start": self.calculation.diis_start,
                "diis_max_vectors": self.calculation.diis_max_vectors,
                "error_metric": self.calculation.error_metric,
            },
            "output": {
                "save_json": self.output.save_json,
                "save_text": self.output.save_text,
                "save_log": self.output.save_log,
                "save_matrices": self.output.save_matrices,
                "log_level": self.output.log_level,
                "json_indent": self.output.json_indent,
                "output_prefix": self.output.output_prefix,
            },
        }

        Path(filepath).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    def __repr__(self) -> str:
        """Return string representation of configuration.

        :returns: Configuration summary.
        :rtype: str
        """
        return (
            f"Configuration(\n"
            f"  output_dir={self.paths.output_dir},\n"
            f"  max_scf_iterations={self.calculation.max_scf_iterations},\n"
            f"  scf_convergence_threshold={self.calculation.scf_convergence_threshold},\n"
            f"  log_level={self.output.log_level}\n"
            f")"
        )
