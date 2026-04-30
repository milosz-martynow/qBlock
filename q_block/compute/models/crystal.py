"""Crystalline atomic system container.

This module defines the :class:`Crystal` class, a minimal placeholder
for crystalline systems that extends :class:`compute.models.atomic_system.AtomicSystem`.
"""

from typing import Optional

from q_block.compute.environment.io.input_data import InputData
from q_block.compute.models.atomic_system import AtomicSystem


class Crystal(AtomicSystem):
    """Container for atoms forming a crystalline system.

    This class is a minimal specialisation of
    :class:`compute.atomic_system.AtomicSystem` intended to represent
    crystals or periodic atomic arrangements. At this stage it does not
    introduce any additional behaviour beyond its base class.

    :param input_data: Optional :class:`InputData` instance describing
        the atoms in the crystalline system.
    :type input_data: Optional[InputData]
    """

    def __init__(self, input_data: Optional[InputData] = None) -> None:
        super().__init__(input_data=input_data)
