r"""
TODO: should this not live in constants.numerical?

Abstract exchange-correlation functional base class.

This module defines the :class:`ExchangeCorrelationFunctional` abstract
base class that all exchange-correlation functionals must inherit from.

Every functional must implement methods to evaluate:

* :math:`\varepsilon_{xc}(\rho)` — the exchange-correlation energy
  density per electron.
* :math:`v_{xc}(\rho) = \partial(\rho\,\varepsilon_{xc})/\partial\rho`
  — the exchange-correlation potential (functional derivative w.r.t.
  the density).

Functional types are encoded via the :attr:`functional_type` attribute:

* ``"lda"`` — depends only on the local density :math:`\rho`.
* ``"gga"`` — depends on :math:`\rho` and :math:`|\nabla\rho|`.
* ``"hybrid"`` — includes a fraction of exact (Hartree-Fock) exchange.

Classes
-------
ExchangeCorrelationFunctional
    Abstract base class for all XC functionals.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExchangeCorrelationFunctional(ABC):
    r"""Abstract exchange-correlation functional.

    Defines the interface that all XC functionals must provide for use
    in Kohn-Sham DFT calculations.

    Subclasses must implement :meth:`compute_exc_vxc`, which returns
    the energy density and potential for each grid point.

    :param name: Human-readable functional name (e.g. ``"SVWN"``,
        ``"PBE"``, ``"B3LYP"``).
    :type name: str
    :param functional_type: Category of the functional.
        One of ``"lda"``, ``"gga"``, ``"hybrid"``.
    :type functional_type: str

    Attributes
    ----------
    name : str
        Functional name.
    functional_type : str
        Functional category.
    exact_exchange_fraction : float
        Fraction of exact (HF) exchange mixed in.  Zero for pure
        LDA/GGA functionals, non-zero for hybrids.
    """

    _VALID_TYPES = ("lda", "gga", "hybrid")

    def __init__(
        self,
        name: str,
        functional_type: str,
        exact_exchange_fraction: float = 0.0,
    ) -> None:
        if functional_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown functional_type {functional_type!r}; "
                f"supported: {self._VALID_TYPES}."
            )
        self.name: str = name
        self.functional_type: str = functional_type
        self.exact_exchange_fraction: float = exact_exchange_fraction

    @property
    def needs_gradient(self) -> bool:
        """Whether this functional requires the density gradient.

        :returns: ``True`` for GGA and hybrid functionals.
        :rtype: bool
        """
        return self.functional_type in ("gga", "hybrid")

    @abstractmethod
    def compute_exc_vxc(
        self,
        rho_alpha: np.ndarray,
        rho_beta: np.ndarray,
        gamma_aa: Optional[np.ndarray] = None,
        gamma_ab: Optional[np.ndarray] = None,
        gamma_bb: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        r"""Compute XC energy density, potentials and gamma derivatives.

        Returns six arrays needed to build the full KS Fock matrix,
        including the GGA gradient correction to the XC potential matrix.

        For LDA functionals the last three arrays are all-zero.  For
        GGA/hybrid functionals they carry the derivative of the XC
        energy density with respect to the spin-resolved gradient
        invariants :math:`\gamma_{\sigma\sigma'}`:

        .. math::

            h_\alpha = \frac{\partial(\rho\,\varepsilon_{xc})}
                            {\partial \gamma_{\alpha\alpha}}, \quad
            h_{ab}   = \frac{\partial(\rho\,\varepsilon_{xc})}
                            {\partial \gamma_{\alpha\beta}}, \quad
            h_\beta  = \frac{\partial(\rho\,\varepsilon_{xc})}
                            {\partial \gamma_{\beta\beta}}.

        These are used by the KS solver to build the GGA correction:

        .. math::

            \Delta V^{xc,\sigma}_{\mu\nu}
                = 2 \sum_g w_g
                  \bigl(
                      h_\sigma(\mathbf{r}_g)\,
                      \nabla\phi_\mu \cdot \nabla\phi_\nu
                    + h_{ab}(\mathbf{r}_g)\,
                      \nabla\phi_\mu \cdot \nabla\phi_\nu
                  \bigr)

        :param rho_alpha: Alpha-spin density at each grid point,
            shape ``(n_points,)``.
        :type rho_alpha: np.ndarray
        :param rho_beta: Beta-spin density at each grid point,
            shape ``(n_points,)``.
        :type rho_beta: np.ndarray
        :param gamma_aa: :math:`|\nabla\rho_\alpha|^2` at each
            grid point (required for GGA/hybrid).
        :type gamma_aa: Optional[np.ndarray]
        :param gamma_ab: :math:`\nabla\rho_\alpha \cdot
            \nabla\rho_\beta` at each grid point.
        :type gamma_ab: Optional[np.ndarray]
        :param gamma_bb: :math:`|\nabla\rho_\beta|^2` at each
            grid point.
        :type gamma_bb: Optional[np.ndarray]
        :returns: ``(exc, vxc_alpha, vxc_beta, h_alpha, h_ab, h_beta)``
            — energy density per electron, per-spin local potentials,
            and gamma derivatives for the GGA correction.
            Each array has shape ``(n_points,)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray]
        """

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"type={self.functional_type!r}, "
            f"exact_exchange={self.exact_exchange_fraction:.2f})"
        )
