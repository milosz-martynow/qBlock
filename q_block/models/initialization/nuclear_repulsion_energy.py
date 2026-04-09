"""Nuclear repulsion energy computation.

This module computes the classical Coulomb repulsion energy between nuclei:

.. math::

    E_{nuc} = \\sum_{A < B}^{N_{atoms}} \\frac{Z_A \\cdot Z_B}{|\\mathbf{R}_A - \\mathbf{R}_B|}

This is a **scalar constant** that depends only on the molecular geometry
(nuclear charges and positions). It does **not** involve any basis functions
or electronic integrals, making it suitable for computation during the
initialization phase.

Classes
-------
NuclearRepulsionEnergy
    Computes and stores the nuclear repulsion energy for a molecular system.
"""

import math
from typing import List, Tuple

from q_block.models.molecule import Molecule


class NuclearRepulsionEnergy:
    """Compute the classical nuclear-nuclear Coulomb repulsion energy.

    The nuclear repulsion energy is the sum of pairwise Coulomb repulsions
    between all nuclei in the molecule:

    .. math::

        E_{nuc} = \\sum_{A < B}^{N_{atoms}} \\frac{Z_A \\cdot Z_B}{R_{AB}}

    where:
    - :math:`Z_A`, :math:`Z_B` are nuclear charges (atomic numbers)
    - :math:`R_{AB} = |\\mathbf{R}_A - \\mathbf{R}_B|` is the internuclear
      distance in Bohr

    This energy is always **positive** (repulsive) and is added to the
    electronic energy at the end of the SCF procedure to obtain the total
    energy:

    .. math::

        E_{total} = E_{elec} + E_{nuc}

    **Important:** Coordinates must be in Bohr (atomic units) before
    computing this energy. Call :meth:`Molecule.to_bohr` if coordinates
    are in Ångström.

    :param molecule: Molecular system with atomic positions in Bohr.
    :type molecule: Molecule

    Attributes
    ----------
    molecule : Molecule
        The molecular system.
    nuclei : List[Tuple[int, Tuple[float, float, float]]]
        List of (charge, position) tuples for each nucleus.
    n_atoms : int
        Number of atoms in the molecule.
    energy : float
        The computed nuclear repulsion energy in Hartree.

    Examples
    --------
    >>> from q_block.models.molecule import Molecule
    >>> from q_block.models.initialization import NuclearRepulsionEnergy
    >>> # H2 molecule at ~0.74 Å bond length (converted to Bohr)
    >>> mol = Molecule(...)
    >>> mol.to_bohr()
    >>> nuc_rep = NuclearRepulsionEnergy(mol)
    >>> print(f"E_nuc = {nuc_rep.energy:.6f} Hartree")
    E_nuc = 0.713776 Hartree
    """

    def __init__(self, molecule: Molecule) -> None:
        self.molecule: Molecule = molecule

        # Extract nuclear data: (Z, (x, y, z)) for each atom
        self.nuclei: List[Tuple[int, Tuple[float, float, float]]] = []
        for atom in self.molecule.atoms:
            Z = atom.atomic_number
            if atom.coordinates is None:
                raise ValueError(
                    f"Atom {atom!r} has no coordinates. "
                    f"Cannot compute nuclear repulsion energy."
                )
            pos = (
                atom.coordinates.x,
                atom.coordinates.y,
                atom.coordinates.z,
            )
            self.nuclei.append((Z, pos))

        self.n_atoms: int = len(self.nuclei)

        # Compute the nuclear repulsion energy
        self.energy: float = self._compute_energy()

    def _compute_energy(self) -> float:
        """Compute the nuclear repulsion energy.

        Iterates over all unique pairs of nuclei (A < B) and sums the
        Coulomb repulsion contributions.

        :returns: Nuclear repulsion energy in Hartree.
        :rtype: float
        """
        e_nuc = 0.0

        for a in range(self.n_atoms):
            Z_a, R_a = self.nuclei[a]
            for b in range(a + 1, self.n_atoms):
                Z_b, R_b = self.nuclei[b]

                # Internuclear distance
                R_ab = math.sqrt(
                    (R_a[0] - R_b[0]) ** 2
                    + (R_a[1] - R_b[1]) ** 2
                    + (R_a[2] - R_b[2]) ** 2
                )

                if R_ab < 1e-10:
                    raise ValueError(
                        f"Nuclei {a} and {b} are at the same position. "
                        f"Cannot compute nuclear repulsion energy."
                    )

                e_nuc += Z_a * Z_b / R_ab

        return e_nuc

    @staticmethod
    def pairwise_energy(
        Z_a: int,
        R_a: Tuple[float, float, float],
        Z_b: int,
        R_b: Tuple[float, float, float],
    ) -> float:
        """Compute pairwise nuclear repulsion between two nuclei.

        .. math::

            E_{AB} = \\frac{Z_A \\cdot Z_B}{R_{AB}}

        :param Z_a: Atomic number of nucleus A.
        :type Z_a: int
        :param R_a: Position of nucleus A (x, y, z) in Bohr.
        :type R_a: Tuple[float, float, float]
        :param Z_b: Atomic number of nucleus B.
        :type Z_b: int
        :param R_b: Position of nucleus B (x, y, z) in Bohr.
        :type R_b: Tuple[float, float, float]

        :returns: Pairwise repulsion energy in Hartree.
        :rtype: float
        """
        R_ab = math.sqrt(
            (R_a[0] - R_b[0]) ** 2
            + (R_a[1] - R_b[1]) ** 2
            + (R_a[2] - R_b[2]) ** 2
        )

        if R_ab < 1e-10:
            raise ValueError("Nuclei are at the same position.")

        return Z_a * Z_b / R_ab

    def __repr__(self) -> str:
        """String representation with atom count and energy."""
        return f"NuclearRepulsionEnergy(n_atoms={self.n_atoms}, energy={self.energy:.6f})"

    def __float__(self) -> float:
        """Return the energy as a float.

        Enables ``float(nuc_rep)`` conversion.

        :returns: Nuclear repulsion energy in Hartree.
        :rtype: float
        """
        return self.energy
