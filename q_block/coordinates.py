"""
Coordinates module for atom representations.

Defines abstract and concrete classes for spatial coordinates.
"""

from typing import Tuple

class Coordinates:
    """
    Abstract base class for coordinates representation.

    This class provides a generic interface for 3D coordinates in any coordinate system.
    Subclasses must implement the protected methods _coordinate_1, _coordinate_2, _coordinate_3,
    which are used internally by as_tuple().
    """
    def as_tuple(self) -> tuple[float, float, float]:
        """
        Return coordinates as a tuple (coordinate_1, coordinate_2, coordinate_3).

        :return: Tuple of three coordinates in the order defined by the coordinate system.
        :rtype: tuple[float, float, float]
        """
        return (self._coordinate_1(), self._coordinate_2(), self._coordinate_3())

    def _coordinate_1(self) -> float:
        """
        Return the first coordinate (system-dependent).

        :return: The first coordinate value.
        :rtype: float
        """
        raise NotImplementedError("Subclasses must implement _coordinate_1.")

    def _coordinate_2(self) -> float:
        """
        Return the second coordinate (system-dependent).

        :return: The second coordinate value.
        :rtype: float
        """
        raise NotImplementedError("Subclasses must implement _coordinate_2.")

    def _coordinate_3(self) -> float:
        """
        Return the third coordinate (system-dependent).

        :return: The third coordinate value.
        :rtype: float
        """
        raise NotImplementedError("Subclasses must implement _coordinate_3.")

class CartesianCoordinates(Coordinates):
    """
    Represents 3D cartesian coordinates (x, y, z) in Angstroms.

    :param x: X coordinate (Angstrom).
    :type x: float
    :param y: Y coordinate (Angstrom).
    :type y: float
    :param z: Z coordinate (Angstrom).
    :type z: float
    """
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initialize CartesianCoordinates with x, y, z values.

        :param x: X coordinate (Angstrom).
        :type x: float
        :param y: Y coordinate (Angstrom).
        :type y: float
        :param z: Z coordinate (Angstrom).
        :type z: float
        """
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def _coordinate_1(self) -> float:
        """
        Return the x coordinate.

        :return: The x coordinate value.
        :rtype: float
        """
        return self.x

    def _coordinate_2(self) -> float:
        """
        Return the y coordinate.

        :return: The y coordinate value.
        :rtype: float
        """
        return self.y

    def _coordinate_3(self) -> float:
        """
        Return the z coordinate.

        :return: The z coordinate value.
        :rtype: float
        """
        return self.z

    def __repr__(self) -> str:
        """
        Return a string representation of the CartesianCoordinates object.

        :return: String representation in the form 'CartesianCoordinates(x=..., y=..., z=...)'.
        :rtype: str
        """
        return f"CartesianCoordinates(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"
