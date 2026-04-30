"""Test templates for Hartree-Fock and DFT validation tests.

This module provides reusable test templates that eliminate code
duplication across RHF, ROHF, UHF, RKS and UKS validation test files.

Each template function returns a test function that can be used directly
by pytest with the appropriate fixtures.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

from q_block.tests.validation.utils import ABS_TOL_EV, DFT_ABS_TOL_EV, HARTREE_TO_EV

# ---------------------------------------------------------------------------
# Atom test templates
# ---------------------------------------------------------------------------


def make_atom_scf_converged_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
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
        ref = entry["energies"]["hf"][0]["value"]
        assert homo_ev == pytest.approx(ref, abs=ABS_TOL_EV), (
            f"Atom {entry['symbol']}: {method_name} Koopmans IE {homo_ev:.3f} eV, "
            f"reference {ref:.3f} eV "
            f"(diff {abs(homo_ev - ref):.3f} eV, tolerance {ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO agrees with "
        f"``energies['hf'][0]['value']``."
    )
    return test_func


# ---------------------------------------------------------------------------
# Molecule test templates
# ---------------------------------------------------------------------------


def make_molecule_scf_converged_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks SCF convergence for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        assert (
            hf.converged
        ), f"{method_name} SCF did not converge for {entry['formula']}."

    test_func.__doc__ = (
        f"{method_name} SCF converges for every open-shell molecule."
        if method_name != "RHF"
        else f"{method_name} SCF converges for every closed-shell molecule."
    )
    return test_func


def make_molecule_total_energy_negative_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], Any]], None]:
    """Create a test function that checks total energy is negative for molecules.

    :param method_name: Name of the method (e.g., "RHF", "ROHF", "UHF").
    :type method_name: str
    :returns: Test function that accepts a fixture returning ``(entry, hf)``.
    :rtype: Callable[[Tuple[Dict[str, Any], Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], Any]) -> None:
        entry, hf = result
        assert (
            hf.e_total < 0.0
        ), f"{entry['formula']}: total energy {hf.e_total:.6f} Ha should be negative."

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
        assert (
            homo_ev > 0.0
        ), f"{entry['formula']}: {method_name} Koopmans IE {homo_ev:.3f} eV should be positive."

    test_func.__doc__ = f"Koopmans IE from the {method_name} HOMO is positive (HOMO is a bound orbital)."
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
        ref = entry["energies"]["hf"][0]["value"]
        assert homo_ev == pytest.approx(ref, abs=ABS_TOL_EV), (
            f"{entry['formula']}: {method_name} Koopmans IE {homo_ev:.3f} eV, "
            f"reference {ref:.3f} eV "
            f"(diff {abs(homo_ev - ref):.3f} eV, tolerance {ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = f"Koopmans IE from the {method_name} HOMO agrees with ``energies['hf'][0]['value']`` within ABS_TOL_EV."
    return test_func


# ---------------------------------------------------------------------------
# DFT (Kohn-Sham) atom test templates
# ---------------------------------------------------------------------------


def make_dft_atom_scf_converged_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT SCF convergence for atoms.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert ks.converged, (
            f"{method_name}-{func_name} SCF did not converge for atom "
            f"{entry['symbol']} (Z={entry['n_electrons']})."
        )

    test_func.__doc__ = f"{method_name} SCF converges for every atom."
    return test_func


def make_dft_atom_total_energy_negative_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT total energy is negative for atoms.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert ks.e_total < 0.0, (
            f"Atom {entry['symbol']}: {method_name}-{func_name} total energy "
            f"{ks.e_total:.6f} Ha should be negative."
        )

    test_func.__doc__ = f"{method_name} total energy is negative for every atom."
    return test_func


def make_dft_atom_total_energy_finite_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT total energy is finite for atoms.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert np.isfinite(ks.e_total), (
            f"Atom {entry['symbol']}: {method_name}-{func_name} produced "
            f"non-finite energy: {ks.e_total}"
        )

    test_func.__doc__ = f"{method_name} total energy is finite for every atom."
    return test_func


def make_dft_atom_homo_ie_positive_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT Koopmans IE is positive for atoms.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``ks`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(ks)[homo_idx] * HARTREE_TO_EV
        assert homo_ev > 0.0, (
            f"Atom {entry['symbol']}: {method_name}-{func_name} Koopmans IE "
            f"{homo_ev:.3f} eV should be positive."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO is positive "
        f"(HOMO is a bound orbital)."
    )
    return test_func


# ---------------------------------------------------------------------------
# DFT (Kohn-Sham) molecule test templates
# ---------------------------------------------------------------------------


def make_dft_molecule_scf_converged_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT SCF convergence for molecules.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert ks.converged, (
            f"{method_name}-{func_name} SCF did not converge for "
            f"{entry['formula']}."
        )

    test_func.__doc__ = f"{method_name} SCF converges for every molecule."
    return test_func


def make_dft_molecule_total_energy_negative_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT total energy is negative for molecules.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert ks.e_total < 0.0, (
            f"{entry['formula']}: {method_name}-{func_name} total energy "
            f"{ks.e_total:.6f} Ha should be negative."
        )

    test_func.__doc__ = f"{method_name} total energy is negative for every molecule."
    return test_func


def make_dft_molecule_total_energy_finite_test(
    method_name: str,
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT total energy is finite for molecules.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        assert np.isfinite(ks.e_total), (
            f"{entry['formula']}: {method_name}-{func_name} produced "
            f"non-finite energy: {ks.e_total}"
        )

    test_func.__doc__ = f"{method_name} total energy is finite for every molecule."
    return test_func


def make_dft_molecule_homo_ie_positive_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT Koopmans IE is positive for molecules.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``ks`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(ks)[homo_idx] * HARTREE_TO_EV
        assert homo_ev > 0.0, (
            f"{entry['formula']}: {method_name}-{func_name} Koopmans IE "
            f"{homo_ev:.3f} eV should be positive."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO is positive "
        f"(HOMO is a bound orbital)."
    )
    return test_func


# ---------------------------------------------------------------------------
# DFT (Kohn-Sham) Koopmans IE comparison templates
# ---------------------------------------------------------------------------


def make_dft_atom_koopmans_ie_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT Koopmans IE vs experimental IE for atoms.

    Compares the DFT HOMO eigenvalue against ``energies['experiment'][0]['value']``
    using the ``DFT_ABS_TOL_EV`` tolerance.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``ks`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(ks)[homo_idx] * HARTREE_TO_EV
        ref = entry["energies"]["experiment"][0]["value"]
        assert homo_ev == pytest.approx(ref, abs=DFT_ABS_TOL_EV), (
            f"Atom {entry['symbol']}: {method_name}-{func_name} Koopmans IE "
            f"{homo_ev:.3f} eV, experimental {ref:.3f} eV "
            f"(diff {abs(homo_ev - ref):.3f} eV, tolerance {DFT_ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO agrees with "
        f"``energies['experiment'][0]['value']`` within DFT_ABS_TOL_EV."
    )
    return test_func


def make_dft_molecule_koopmans_ie_test(
    method_name: str,
    homo_idx_func: Callable[[Dict[str, Any]], int],
    epsilon_func: Callable[[Any], Any],
) -> Callable[[Tuple[Dict[str, Any], str, Any]], None]:
    """Create a test that checks DFT Koopmans IE vs experimental IE for molecules.

    Compares the DFT HOMO eigenvalue against ``energies['experiment'][0]['value']``
    using the ``DFT_ABS_TOL_EV`` tolerance.

    :param method_name: Name of the method (e.g., "RKS", "UKS").
    :type method_name: str
    :param homo_idx_func: Function that takes ``entry`` and returns HOMO index.
    :type homo_idx_func: Callable[[Dict[str, Any]], int]
    :param epsilon_func: Function that takes ``ks`` and returns eigenvalue array.
    :type epsilon_func: Callable[[Any], Any]
    :returns: Test function that accepts a fixture returning
        ``(entry, func_name, ks)``.
    :rtype: Callable[[Tuple[Dict[str, Any], str, Any]], None]
    """

    def test_func(result: Tuple[Dict[str, Any], str, Any]) -> None:
        entry, func_name, ks = result
        homo_idx = homo_idx_func(entry)
        homo_ev = -epsilon_func(ks)[homo_idx] * HARTREE_TO_EV
        ref = entry["energies"]["experiment"][0]["value"]
        assert homo_ev == pytest.approx(ref, abs=DFT_ABS_TOL_EV), (
            f"{entry['formula']}: {method_name}-{func_name} Koopmans IE "
            f"{homo_ev:.3f} eV, experimental {ref:.3f} eV "
            f"(diff {abs(homo_ev - ref):.3f} eV, tolerance {DFT_ABS_TOL_EV} eV)."
        )

    test_func.__doc__ = (
        f"Koopmans IE from the {method_name} HOMO agrees with "
        f"``energies['experiment'][0]['value']`` within DFT_ABS_TOL_EV."
    )
    return test_func
