"""Unit tests for q_block.models.crystal module.

Tests cover:
- Crystal construction as AtomicSystem subclass
- Default initialization

All tests use pytest with parametrize, no test classes.
"""

from q_block.models.atomic_system import AtomicSystem
from q_block.models.crystal import Crystal


def test_crystal_is_atomic_system_subclass() -> None:
    """Verify Crystal inherits from AtomicSystem."""
    assert issubclass(Crystal, AtomicSystem)


def test_crystal_default_construction() -> None:
    """Verify Crystal can be constructed without arguments."""
    crystal = Crystal()
    assert crystal.input_data is not None
