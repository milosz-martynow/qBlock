"""Shared utilities for all Koopmans' theorem validation tests."""

from pathlib import Path
from typing import Dict, List, Tuple

from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.systems.molecule import Molecule
from q_block.theory.initialization.nuclear_repulsion_energy import NuclearRepulsionEnergy

HARTREE_TO_EV: float = 27.211386
"""Conversion factor: 1 Hartree = 27.211386 eV."""

ABS_TOL_EV: float = 3.0
"""Absolute tolerance (eV) for Koopmans IE comparison.

3-21G Koopmans IEs typically agree with near-basis-set-limit HF reference
values within 0.5–2 eV for first-row systems and up to 3 eV for heavier
elements.  This tolerance catches implementation bugs (wrong sign, wrong
units, wrong orbital index) while remaining robust to basis-set effects.
"""

_BASIS_CACHE: Dict[str, Pople] = {}


def _get_basis(filename: str) -> Pople:
    """Return a cached :class:`Pople` basis set loaded from *filename*.

    :param filename: Basis-set file name, e.g. ``"6-311++Gss.gbs"``.
    """
    if filename not in _BASIS_CACHE:
        _BASIS_CACHE[filename] = Pople(
            filepath=str(Path("data/basis_set/gto_gaussian_format") / filename)
        )
    return _BASIS_CACHE[filename]


def _build_from_geometry(
    geometry: List[Dict],
    multiplicity: int,
    basis_set_filename: str = "3-21G.gbs",
) -> Tuple:
    """Build a Molecule from a geometry list and return SCF inputs.

    :param geometry: List of ``{"symbol", "x", "y", "z"}`` dicts (Å).
    :param multiplicity: Spin multiplicity 2S+1.
    :param basis_set_filename: Basis-set file name (e.g. ``"6-311++Gss.gbs"``).
    :returns: ``(cgtos, nuclei, e_nuclear)`` ready for HF constructors.
    """
    basis = _get_basis(basis_set_filename)
    inp = InputData()
    inp.from_script(
        atom_data=[
            [atom["symbol"], atom["x"], atom["y"], atom["z"], basis]
            for atom in geometry
        ]
    )
    mol = Molecule(input_data=inp, multiplicity=multiplicity)
    mol.to_bohr()
    mol.make_contracted_gaussian_type_orbital()
    cgtos = mol.contracted_gaussian_type_orbitals
    nuclei = [
        (a.atomic_number, (a.coordinates.x, a.coordinates.y, a.coordinates.z))
        for a in mol.atoms
    ]
    e_nuclear = NuclearRepulsionEnergy(mol).energy
    return cgtos, nuclei, e_nuclear
