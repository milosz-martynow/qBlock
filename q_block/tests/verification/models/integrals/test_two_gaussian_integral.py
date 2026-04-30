"""Unit tests for compute.models.integrals.two_gaussian_integral module.

Tests cover:
- TwoGaussianIntegral abstract base class interface
- Cannot instantiate base TwoGaussianIntegral directly

All tests use pytest with parametrize, no test classes.
"""

import pytest

from q_block.compute.models.integrals.two_gaussian_integral import TwoGaussianIntegral


def test_two_gaussian_integral_cannot_be_instantiated() -> None:
    """Verify TwoGaussianIntegral ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        TwoGaussianIntegral()
