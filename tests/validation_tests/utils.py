"""Shared utilities for all Koopmans' theorem validation tests."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from compute.environment.io.basis_set import Pople
from compute.environment.io.input_data import InputData
from compute.models.initialization.nuclear_repulsion_energy import (
    NuclearRepulsionEnergy,
)
from compute.models.molecule import Molecule

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


def _get_basis(filepath: Union[Path, str]) -> Pople:
    """Return a cached :class:`Pople` basis set loaded from *filepath*.

    :param filepath: Full path to basis-set file, e.g. ``Path("compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs")``.
    """
    filepath_str = str(filepath)
    if filepath_str not in _BASIS_CACHE:
        _BASIS_CACHE[filepath_str] = Pople(filepath=filepath_str)
    return _BASIS_CACHE[filepath_str]


def _build_from_geometry(
    geometry: List[Dict],
    multiplicity: int,
    basis_set_filename: Union[Path, str] = Path(
        "compute/environment/constants/numerical/basis_set/pople/3-21G.gbs"
    ),
) -> Tuple:
    """Build a Molecule from a geometry list and return SCF inputs.

    :param geometry: List of ``{"symbol", "x", "y", "z"}`` dicts (Å).
    :param multiplicity: Spin multiplicity 2S+1.
    :param basis_set_filename: Full path to basis-set file (e.g. ``Path("compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs")``).
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
