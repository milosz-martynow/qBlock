from q_block.atom import Atom
from q_block.basis_set_pople import parse_gaussian_basis

gto_gbs = "data/basis_set/gto_gaussian_format/3-21G.gbs"

if __name__ == "__main__":

    basis = parse_gaussian_basis(filepath=gto_gbs)
    ATOM_NUMBER = 6

    atom = Atom(Z=ATOM_NUMBER, basis_set=basis)
    atom.fill_occupancy()
    atom.populate_spinorbitals_with_gto()
