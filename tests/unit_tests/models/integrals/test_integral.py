"""Unit tests for q_block.models.integrals.integral module.

Tests cover:
- Integral abstract base class interface
- Cannot instantiate base Integral directly

All tests use pytest with parametrize, no test classes.
"""

import pytest

from q_block.models.integrals.integral import Integral


def test_integral_cannot_be_instantiated() -> None:
    """Verify Integral ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Integral()
