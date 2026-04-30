r"""Configuration reader for qBlock calculations.

This module reads input configuration for qBlock from a plain-text
configuration file.  The default file name is ``.qblock.config``;
a documented template is shipped as ``.qblock.config.example``.

Recognised Configuration Keys
------------------------------

basis_set
    Global basis set name applied to every atom that does not specify
    its own in the geometry block (e.g. ``STO-3G``, ``3-21G``,
    ``6-31G``, ``6-311++G**``).  Default: ``STO-3G``.

charge
    Total system charge (integer).  Default: ``0``.

multiplicity
    Spin multiplicity :math:`2S+1` (integer).  Default: ``1``.

geometry_file
    Path to an XYZ geometry file.  Used when geometry is not defined
    inline.  Mutually exclusive with the ``$geometry`` block.

max_scf_iterations
    Maximum number of SCF iterations (integer).  Default: ``100``.

scf_convergence_threshold
    Energy / density convergence threshold (float).
    Default: ``1e-8``.

diis_start
    Iteration at which DIIS extrapolation begins, 0-indexed (integer).
    Default: ``1``.

diis_max_vectors
    Maximum DIIS subspace size (integer).  Default: ``6``.

error_metric
    Convergence error metric.  Accepted: ``rms``, ``max``.
    Default: ``rms``.

output_dir
    Directory for calculation output files.  Default: ``output``.

log_level
    Logging verbosity.  Accepted: ``DEBUG``, ``INFO``, ``WARNING``,
    ``ERROR``.  Default: ``INFO``.

save_json
    Write JSON output (``true`` / ``false``).  Default: ``true``.

save_text
    Write plain-text summary output (``true`` / ``false``).
    Default: ``true``.

Geometry Block
--------------

Inline geometry is enclosed between ``$geometry`` and ``$end``
markers.  Each line describes one atom::

    AtomSymbol  X  Y  Z  [basis_set]  [charge]

Per-atom *basis_set* overrides the global ``basis_set`` value.
Per-atom *charge* is the formal atomic charge (defaults to ``0``
when omitted).

File Format Example
-------------------

::

    # Global settings
    basis_set = STO-3G
    charge = 0
    multiplicity = 1

    # Calculation parameters
    max_scf_iterations = 100
    scf_convergence_threshold = 1e-8

    # Inline geometry
    $geometry
    H  0.0  0.0  0.0
    H  0.0  0.0  0.74
    $end

Usage
-----

1. **Load from file**::

    from q_block.compute.environment.configuration import Configuration

    config = Configuration.from_file("my_calculation.qblock.config")

2. **Load from default location**::

    config = Configuration.from_file()
    # Searches for .qblock.config in the current directory

3. **Programmatic creation**::

    config = Configuration(
        basis_set="3-21G",
        charge=0,
        multiplicity=1,
        geometry=[
            ["H", 0.0, 0.0, 0.0],
            ["H", 0.0, 0.0, 0.74],
        ],
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Mapping of recognised keys to (type, default_value).
_KEY_DEFAULTS: Dict[str, Tuple[type, Any]] = {
    "basis_set": (str, "STO-3G"),
    "charge": (int, 0),
    "multiplicity": (int, 1),
    "geometry_file": (str, None),
    "max_scf_iterations": (int, 100),
    "scf_convergence_threshold": (float, 1e-8),
    "diis_start": (int, 1),
    "diis_max_vectors": (int, 6),
    "error_metric": (str, "rms"),
    "output_dir": (str, "output"),
    "log_level": (str, "INFO"),
    "save_json": (bool, True),
    "save_text": (bool, True),
}


def _cast_value(key: str, raw: str) -> Any:
    """Cast a raw string value to the expected type for *key*.

    :param key: Recognised configuration key.
    :type key: str
    :param raw: Raw string value from the configuration file.
    :type raw: str
    :returns: Value cast to the appropriate Python type.
    :rtype: Any
    :raises ValueError: If *key* is unknown or *raw* cannot be
        converted.
    """
    if key not in _KEY_DEFAULTS:
        raise ValueError(f"Unknown configuration key: {key!r}")

    target_type: type = _KEY_DEFAULTS[key][0]

    if target_type is bool:
        return raw.strip().lower() in ("true", "1", "yes")
    if target_type is int:
        return int(raw.strip())
    if target_type is float:
        return float(raw.strip())
    return raw.strip()


def _parse_geometry_line(
    line: str,
    global_basis_set: Optional[str],
) -> List[Any]:
    """Parse a single geometry line into an atom entry list.

    Expected format::

        AtomSymbol  X  Y  Z  [basis_set]  [charge]

    :param line: Single geometry line.
    :type line: str
    :param global_basis_set: Fallback basis set name.
    :type global_basis_set: Optional[str]
    :returns: List ``[symbol, x, y, z]``, or
        ``[symbol, x, y, z, basis_set_name]``, or
        ``[symbol, x, y, z, basis_set_name, charge]``.
    :rtype: List[Any]
    :raises ValueError: If the line cannot be parsed.
    """
    parts: List[str] = line.split()
    if len(parts) < 4:
        raise ValueError(
            f"Geometry line must have at least 4 fields "
            f"(Symbol X Y Z); got: {line!r}"
        )

    symbol: str = parts[0]

    try:
        x: float = float(parts[1])
        y: float = float(parts[2])
        z: float = float(parts[3])
    except ValueError as exc:
        raise ValueError(
            f"Coordinates must be numeric; got: {line!r}"
        ) from exc

    basis_set: Optional[str] = (
        parts[4] if len(parts) > 4 else global_basis_set
    )

    if len(parts) > 5:
        try:
            charge: int = int(parts[5])
        except ValueError as exc:
            raise ValueError(
                f"Per-atom charge must be an integer; "
                f"got: {parts[5]!r}"
            ) from exc
        return [symbol, x, y, z, basis_set, charge]

    if basis_set is not None:
        return [symbol, x, y, z, basis_set]

    return [symbol, x, y, z]


class Configuration:
    """Input configuration container for qBlock calculations.

    Stores all parameters needed to set up and run a qBlock
    calculation.  Can be created programmatically or loaded from
    a plain-text configuration file via :meth:`from_file`.

    :param basis_set: Global basis set name.
    :type basis_set: str
    :param charge: Total system charge.
    :type charge: int
    :param multiplicity: Spin multiplicity (2S+1).
    :type multiplicity: int
    :param geometry_file: Path to an XYZ geometry file.
    :type geometry_file: Optional[str]
    :param geometry: Inline atom entries, each
        ``[symbol, x, y, z]`` or
        ``[symbol, x, y, z, basis_set, charge]``.
    :type geometry: Optional[List[List[Any]]]
    :param max_scf_iterations: Maximum SCF iterations.
    :type max_scf_iterations: int
    :param scf_convergence_threshold: Convergence threshold.
    :type scf_convergence_threshold: float
    :param diis_start: Iteration to start DIIS.
    :type diis_start: int
    :param diis_max_vectors: Maximum DIIS subspace size.
    :type diis_max_vectors: int
    :param error_metric: Convergence error metric
        (``"rms"`` or ``"max"``).
    :type error_metric: str
    :param output_dir: Output directory path.
    :type output_dir: str
    :param log_level: Logging level string.
    :type log_level: str
    :param save_json: Write JSON output.
    :type save_json: bool
    :param save_text: Write text summary output.
    :type save_text: bool

    Attributes
    ----------
    basis_set : str
    charge : int
    multiplicity : int
    geometry_file : Optional[str]
    geometry : List[List[Any]]
    max_scf_iterations : int
    scf_convergence_threshold : float
    diis_start : int
    diis_max_vectors : int
    error_metric : str
    output_dir : str
    log_level : str
    save_json : bool
    save_text : bool
    """

    def __init__(
        self,
        basis_set: str = "STO-3G",
        charge: int = 0,
        multiplicity: int = 1,
        geometry_file: Optional[str] = None,
        geometry: Optional[List[List[Any]]] = None,
        max_scf_iterations: int = 100,
        scf_convergence_threshold: float = 1e-8,
        diis_start: int = 1,
        diis_max_vectors: int = 6,
        error_metric: str = "rms",
        output_dir: str = "output",
        log_level: str = "INFO",
        save_json: bool = True,
        save_text: bool = True,
    ) -> None:
        self.basis_set: str = basis_set
        self.charge: int = charge
        self.multiplicity: int = multiplicity
        self.geometry_file: Optional[str] = geometry_file
        self.geometry: List[List[Any]] = (
            geometry if geometry is not None else []
        )
        self.max_scf_iterations: int = max_scf_iterations
        self.scf_convergence_threshold: float = (
            scf_convergence_threshold
        )
        self.diis_start: int = diis_start
        self.diis_max_vectors: int = diis_max_vectors
        self.error_metric: str = error_metric
        self.output_dir: str = output_dir
        self.log_level: str = log_level
        self.save_json: bool = save_json
        self.save_text: bool = save_text

    # ── Parameter accessors (proper format) ──────────────────────

    def get_basis_set(self) -> str:
        """Return global basis set name.

        :returns: Basis set name (e.g. ``"STO-3G"``).
        :rtype: str
        """
        return self.basis_set

    def get_charge(self) -> int:
        """Return total system charge.

        :returns: System charge.
        :rtype: int
        """
        return self.charge

    def get_multiplicity(self) -> int:
        """Return spin multiplicity (2S+1).

        :returns: Spin multiplicity.
        :rtype: int
        """
        return self.multiplicity

    def get_geometry_file(self) -> Optional[Path]:
        """Return geometry file path as a :class:`Path`.

        :returns: Geometry file path, or ``None`` when inline
            geometry is used.
        :rtype: Optional[Path]
        """
        if self.geometry_file is None:
            return None
        return Path(self.geometry_file)

    def get_geometry(self) -> List[List[Any]]:
        """Return inline geometry entries.

        Each entry is ``[symbol, x, y, z]``,
        ``[symbol, x, y, z, basis_set_name]``, or
        ``[symbol, x, y, z, basis_set_name, charge]``.

        :returns: List of atom entries.
        :rtype: List[List[Any]]
        """
        return self.geometry

    def get_max_scf_iterations(self) -> int:
        """Return maximum SCF iteration count.

        :returns: Maximum iterations.
        :rtype: int
        """
        return self.max_scf_iterations

    def get_scf_convergence_threshold(self) -> float:
        """Return SCF convergence threshold.

        :returns: Convergence threshold.
        :rtype: float
        """
        return self.scf_convergence_threshold

    def get_diis_start(self) -> int:
        """Return DIIS start iteration (0-indexed).

        :returns: DIIS start iteration.
        :rtype: int
        """
        return self.diis_start

    def get_diis_max_vectors(self) -> int:
        """Return maximum DIIS subspace size.

        :returns: Maximum DIIS vectors.
        :rtype: int
        """
        return self.diis_max_vectors

    def get_error_metric(self) -> str:
        """Return convergence error metric name.

        :returns: ``"rms"`` or ``"max"``.
        :rtype: str
        """
        return self.error_metric

    def get_output_dir(self) -> Path:
        """Return output directory as a :class:`Path`.

        :returns: Output directory path.
        :rtype: Path
        """
        return Path(self.output_dir)

    def get_log_level(self) -> str:
        """Return logging level string.

        :returns: Log level (``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``).
        :rtype: str
        """
        return self.log_level

    def get_save_json(self) -> bool:
        """Return whether JSON output should be saved.

        :returns: ``True`` to save JSON output.
        :rtype: bool
        """
        return self.save_json

    def get_save_text(self) -> bool:
        """Return whether text summary output should be saved.

        :returns: ``True`` to save text output.
        :rtype: bool
        """
        return self.save_text

    @classmethod
    def from_file(
        cls,
        filepath: Optional[Union[str, Path]] = None,
    ) -> "Configuration":
        """Load configuration from a plain-text file.

        If *filepath* is ``None``, searches for ``.qblock.config``
        in the current directory.  When no file is found a default
        :class:`Configuration` is returned.

        :param filepath: Path to the configuration file.  When
            ``None``, defaults to ``.qblock.config``.
        :type filepath: Optional[Union[str, Path]]
        :returns: Parsed configuration.
        :rtype: Configuration
        :raises FileNotFoundError: If an explicit *filepath* does
            not exist.
        :raises ValueError: If the file contains syntax errors.
        """
        if filepath is None:
            default_path: Path = Path(".qblock.config")
            if not default_path.exists():
                return cls()
            path: Path = default_path
        else:
            path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        text: str = path.read_text(encoding="utf-8")
        return cls._parse(text)

    @classmethod
    def _parse(cls, text: str) -> "Configuration":
        """Parse configuration from plain-text content.

        :param text: Raw file content.
        :type text: str
        :returns: Parsed configuration instance.
        :rtype: Configuration
        :raises ValueError: On syntax errors.
        """
        settings: Dict[str, Any] = {}
        geometry_lines: List[str] = []
        in_geometry: bool = False

        for line_no, raw_line in enumerate(
            text.splitlines(), start=1
        ):
            line: str = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Geometry block markers
            if line.lower() == "$geometry":
                if in_geometry:
                    raise ValueError(
                        f"Line {line_no}: nested $geometry block"
                    )
                in_geometry = True
                continue

            if line.lower() == "$end":
                if not in_geometry:
                    raise ValueError(
                        f"Line {line_no}: $end without $geometry"
                    )
                in_geometry = False
                continue

            # Inside geometry block
            if in_geometry:
                geometry_lines.append(line)
                continue

            # Key = value pair
            if "=" not in line:
                raise ValueError(
                    f"Line {line_no}: expected 'key = value', "
                    f"got: {raw_line!r}"
                )

            key: str
            value: str
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if not key:
                raise ValueError(
                    f"Line {line_no}: empty key"
                )

            settings[key] = _cast_value(key, value)

        if in_geometry:
            raise ValueError(
                "Unclosed $geometry block (missing $end)"
            )

        # Resolve global basis set for geometry defaults
        global_basis_set: Optional[str] = settings.get(
            "basis_set",
            _KEY_DEFAULTS["basis_set"][1],
        )

        geometry: List[List[Any]] = [
            _parse_geometry_line(gl, global_basis_set)
            for gl in geometry_lines
        ]

        # Build kwargs with defaults for missing keys
        kwargs: Dict[str, Any] = {}
        for key, (_, default) in _KEY_DEFAULTS.items():
            kwargs[key] = settings.get(key, default)

        kwargs["geometry"] = geometry

        return cls(**kwargs)

    def __repr__(self) -> str:
        """Return string representation of configuration.

        :returns: Configuration summary.
        :rtype: str
        """
        geom_desc: str = (
            f"geometry_file={self.geometry_file!r}"
            if self.geometry_file
            else f"n_atoms={len(self.geometry)}"
        )
        return (
            f"Configuration("
            f"basis_set={self.basis_set!r}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multiplicity}, "
            f"{geom_desc}, "
            f"max_scf_iterations={self.max_scf_iterations}"
            f")"
        )
