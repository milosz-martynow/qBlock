r"""Spin-resolved matrix pair container.

Provides the :class:`SpinPair` class, a lightweight ``(alpha, beta)``
tuple subclass used throughout spin-polarised electronic-structure
solvers (Hartree-Fock, DFT, post-HF, etc.).

Classes
-------
SpinPair
    Tuple subclass holding ``(alpha, beta)`` matrix pairs with an
    optional *shared* mode for identical channels.
"""

from typing import Optional, Union

import numpy as np


class SpinPair(tuple):
    r"""Spin-resolved matrix pair (alpha, beta).

    Lightweight container for the ``(alpha, beta)`` matrix pairs that
    appear throughout spin-polarised solvers (Fock matrices,
    density matrices, MO coefficients, orbital energies).

    When *beta* is ``None`` both spin channels share the same
    underlying array object (*shared* mode).  This avoids redundant
    storage for closed-shell RHF and enables computational shortcuts
    — for example, :meth:`HartreeFock._build_fock` skips the second
    exchange-matrix contraction when the density is shared.

    Subclasses :class:`tuple`, so instances pass
    ``isinstance(x, tuple)`` checks and work transparently with
    :class:`~q_block.solvers.diis.DIIS`,
    :func:`~q_block.solvers.diagonalisation.diagonalise_fock`, and
    other routines that expect a tuple of matrices.

    :param alpha: Matrix for the alpha spin channel.
    :type alpha: np.ndarray
    :param beta: Matrix for the beta spin channel.  ``None``
        (default) selects shared mode, where ``.beta`` returns the
        same object as ``.alpha``.
    :type beta: Optional[np.ndarray]
    """

    def __new__(
        cls,
        alpha: np.ndarray,
        beta: Optional[np.ndarray] = None,
    ) -> 'SpinPair':
        return super().__new__(
            cls, (alpha, alpha if beta is None else beta),
        )

    def __init__(
        self,
        alpha: np.ndarray,
        beta: Optional[np.ndarray] = None,
    ) -> None:
        self._shared: bool = beta is None

    # ------ Named access ------------------------------------------------

    @property
    def alpha(self) -> np.ndarray:
        """Alpha-spin matrix."""
        return self[0]

    @property
    def beta(self) -> np.ndarray:
        """Beta-spin matrix (same object as *alpha* when shared)."""
        return self[1]

    @property
    def shared(self) -> bool:
        """``True`` when both channels reference the same matrix."""
        return self._shared

    @property
    def total(self) -> np.ndarray:
        """Element-wise sum ``alpha + beta``.

        Uses ``2 * alpha`` when shared, avoiding a redundant addition.
        """
        if self._shared:
            return 2.0 * self[0]
        return self[0] + self[1]

    # ------ Factory helpers ---------------------------------------------

    @classmethod
    def wrap(cls, obj: Union[tuple, 'SpinPair']) -> 'SpinPair':
        """Ensure *obj* is a :class:`SpinPair`.

        Returns *obj* unchanged if it already is one; otherwise wraps
        the first two elements into a new (non-shared)
        :class:`SpinPair`.

        :param obj: Tuple or SpinPair to wrap.
        :type obj: Union[tuple, SpinPair]
        :returns: SpinPair instance.
        :rtype: SpinPair
        """
        if isinstance(obj, cls):
            return obj
        return cls(obj[0], obj[1])

    # ------ repr --------------------------------------------------------

    def __repr__(self) -> str:
        mode = "shared" if self._shared else "independent"
        shape = self[0].shape if hasattr(self[0], 'shape') else '?'
        return f"SpinPair(shape={shape}, {mode})"
