"""Test templates for Hartree-Fock validation tests.

This module provides reusable test templates that eliminate code
duplication across RHF, ROHF and UHF validation test files.

Each template function returns a test function that can be used directly
by pytest with the appropriate fixtures.
"""

import pytest
from typing import Any, Callable, Dict, Tuple

from tests.validation_tests.utils import ABS_TOL_EV, HARTREE_TO_EV


# ---------------------------------------------------------------------------
# Atom test templates
# ---------------------------------------------------------------------------


def make_atom_scf_converged_test(method_name: str) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks SCF convergence for atoms.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        assert hf.converged, (
            f"{method_name} SCF did not converge for atom {entry['symbol']} "
            f"(Z={entry['n_electrons']})."
        )

    test_func.__doc__ = (
        f"{method_name} SCF converges for every open-shell atom."
        if method_name != "RHF"
        else f"{method_name} SCF converges for every closed-shell atom."
    )
    return test_func


def make_atom_koopmans_ie_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks Koopmans IE vs reference for atoms.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``hf`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(hf)[homo_idx] * HARTREE_TO_EV
        assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
            f"Atom {entry['symbol']}: {method_name} Koopmans IE {homo_ev:.3f} eV, "
            f"reference {entry['hf_ie_eV']:.3f} eV "
            f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO agrees with ``hf_ie_eV``."
    )
    return test_func


# ---------------------------------------------------------------------------
# Molecule test templates
# ---------------------------------------------------------------------------


def make_molecule_scf_converged_test(method_name: str) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks SCF convergence for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        assert hf.converged, (
            f"{method_name} SCF did not converge for {entry['formula']}."
        )

    test_func.__doc__ = (
        f"{method_name} SCF converges for every open-shell molecule."
        if method_name != "RHF"
        else f"{method_name} SCF converges for every closed-shell molecule."
    )
    return test_func


def make_molecule_total_energy_negative_test(method_name: str) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks total energy is negative for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        assert hf.e_total < 0.0, (
            f"{entry['formula']}: total energy {hf.e_total:.6f} Ha should be negative."
        )

    test_func.__doc__ = (
        f"{method_name} total energy is negative for every open-shell molecule."
        if method_name != "RHF"
        else f"{method_name} total energy is negative for every closed-shell molecule."
    )
    return test_func


def make_molecule_homo_ie_positive_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks Koopmans IE is positive for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``hf`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(hf)[homo_idx] * HARTREE_TO_EV
        assert homo_ev > 0.0, (
            f"{entry['formula']}: {method_name} Koopmans IE {homo_ev:.3f} eV should be positive."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO is positive (HOMO is a bound orbital)."
    )
    return test_func


def make_molecule_koopmans_ie_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks Koopmans IE vs reference for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``hf`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(hf)[homo_idx] * HARTREE_TO_EV
        assert homo_ev == pytest.approx(entry["hf_ie_eV"], abs=ABS_TOL_EV), (
            f"{entry['formula']}: {method_name} Koopmans IE {homo_ev:.3f} eV, "
            f"reference {entry['hf_ie_eV']:.3f} eV "
            f"(diff {abs(homo_ev - entry['hf_ie_eV']):.3f} eV, tolerance {ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO agrees with ``hf_ie_eV`` within ABS_TOL_EV."
    )
    return test_func
