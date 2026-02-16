"""Example: Using Contracted Gaussian-Type Orbitals (CGTOs) and Molecular Orbitals.

This example demonstrates how to:
1. Build CGTO basis functions for atoms and molecules
2. Evaluate primitive Gaussians at arbitrary points in space
3. Evaluate contracted basis functions
4. Evaluate molecular orbitals from MO coefficients

The mathematical foundation:

    Primitive Gaussian:
        g_p(r) = N * (x-Rx)^lx * (y-Ry)^ly * (z-Rz)^lz * exp(-α_p * |r-R|²)

    Contracted Basis Function (CGTO):
        φ_μ(r) = Σ_p d_p * g_p(r)

    Molecular Orbital:
        ψ_i(r) = Σ_μ C_μi * φ_μ(r)
"""

from q_block import Atom, Molecule, ContractedGaussianTypeOrbital
from q_block.io.basis_set import Pople
from q_block.io.input_data import InputData
from q_block.io.coordinates import CartesianCoordinates


def main():
    # ══════════════════════════════════════════════════════════════════
    # 1. Load a basis set
    # ══════════════════════════════════════════════════════════════════
    basis = Pople(filepath="data/basis_set/gto_gaussian_format/6-31G.gbs")
    print("Loaded 6-31G basis set\n")

    # ══════════════════════════════════════════════════════════════════
    # 2. Create an atom and build its CGTOs
    # ══════════════════════════════════════════════════════════════════
    hydrogen = Atom(
        atomic_number=1,
        basis_set=basis,
        coordinates=CartesianCoordinates(0.0, 0.0, 0.0),
    )
    hydrogen.make_contracted_gaussian_type_orbital()

    print("=== Hydrogen atom CGTOs ===")
    for i, cgto in enumerate(hydrogen.contracted_gaussian_type_orbitals):
        print(f"  Shell {i}: l={cgto.l}, K={cgto.n_primitives}, "
              f"contributes {cgto.n_functions} basis function(s)")
        print(f"           exponents: {cgto.exponents}")
        print(f"           coefficients: {cgto.contractions}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 3. Evaluate a primitive Gaussian at different points
    # ══════════════════════════════════════════════════════════════════
    cgto = hydrogen.contracted_gaussian_type_orbitals[0]  # First shell (1s)

    # Test points
    origin = CartesianCoordinates(0.0, 0.0, 0.0)
    point_x = CartesianCoordinates(0.5, 0.0, 0.0)
    point_far = CartesianCoordinates(2.0, 0.0, 0.0)

    print("=== Primitive Gaussian evaluation (1s shell, primitive 0) ===")
    # For s-orbital: lx=ly=lz=0
    print(f"  At origin:    {cgto.primitive_gaussian(origin, 0, 0, 0, 0):.6f}")
    print(f"  At (0.5,0,0): {cgto.primitive_gaussian(point_x, 0, 0, 0, 0):.6f}")
    print(f"  At (2.0,0,0): {cgto.primitive_gaussian(point_far, 0, 0, 0, 0):.6f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 4. Evaluate the contracted basis function
    # ══════════════════════════════════════════════════════════════════
    print("=== Contracted basis function evaluation (1s) ===")
    print(f"  At origin:    {cgto.basis_function(origin, 0, 0, 0):.6f}")
    print(f"  At (0.5,0,0): {cgto.basis_function(point_x, 0, 0, 0):.6f}")
    print(f"  At (2.0,0,0): {cgto.basis_function(point_far, 0, 0, 0):.6f}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 5. Create a water molecule and build all CGTOs
    # ══════════════════════════════════════════════════════════════════
    inp = InputData()
    inp.from_script(atom_data=[
        ["O", 0.0000,  0.0000,  0.1173, basis],
        ["H", 0.0000,  0.7572, -0.4692, basis],
        ["H", 0.0000, -0.7572, -0.4692, basis],
    ])
    water = Molecule(input_data=inp)
    water.make_contracted_gaussian_type_orbital()

    print("=== Water molecule CGTOs ===")
    print(f"  Total shells: {len(water.contracted_gaussian_type_orbitals)}")
    print(f"  Total basis functions (n_basis): {water.n_basis}")
    for i, cgto in enumerate(water.contracted_gaussian_type_orbitals):
        shell_label = "spdfghiklm"[cgto.l] if cgto.l < 10 else f"l{cgto.l}"
        print(f"  Shell {i:2d}: atom {cgto.atom_index}, {shell_label}-type, "
              f"K={cgto.n_primitives}, n_functions={cgto.n_functions}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 6. Build angular component list for molecular orbital evaluation
    # ══════════════════════════════════════════════════════════════════
    # Each shell with angular momentum l contributes (2l+1) basis functions.
    # For Cartesian Gaussians:
    #   l=0 (s): (0,0,0)
    #   l=1 (p): (1,0,0), (0,1,0), (0,0,1)
    #   l=2 (d): (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)

    def cartesian_components(l: int):
        """Generate Cartesian angular momentum tuples for given l."""
        components = []
        for lx in range(l, -1, -1):
            for ly in range(l - lx, -1, -1):
                lz = l - lx - ly
                components.append((lx, ly, lz))
        return components

    angular_components = []
    for cgto in water.contracted_gaussian_type_orbitals:
        angular_components.extend(cartesian_components(cgto.l))

    print("=== Angular components for basis functions ===")
    print(f"  Total: {len(angular_components)} (should match n_basis={water.n_basis})")
    print(f"  First 10: {angular_components[:10]}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 7. Evaluate a molecular orbital
    # ══════════════════════════════════════════════════════════════════
    # In real HF calculations, coefficients come from diagonalizing the
    # Fock matrix. Here we use a simple example: a unit vector that
    # selects just the first basis function (the oxygen 1s).

    print("=== Molecular orbital evaluation ===")

    # Example 1: MO that is just the first basis function (O 1s)
    coefficients_1s = [1.0] + [0.0] * (water.n_basis - 1)

    r_test = CartesianCoordinates(0.0, 0.0, 0.1173)  # Near oxygen
    mo_value = water.molecular_orbital(r_test, coefficients_1s, angular_components)
    print(f"  MO (pure O-1s) at oxygen position: {mo_value:.6f}")

    # Example 2: Equal mix of first two basis functions
    coefficients_mix = [0.5, 0.5] + [0.0] * (water.n_basis - 2)
    mo_value_mix = water.molecular_orbital(r_test, coefficients_mix, angular_components)
    print(f"  MO (0.5*bf1 + 0.5*bf2) at oxygen: {mo_value_mix:.6f}")

    # Evaluate along z-axis to see orbital shape
    print("\n  MO (pure O-1s) along z-axis:")
    for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
        r = CartesianCoordinates(0.0, 0.0, z)
        val = water.molecular_orbital(r, coefficients_1s, angular_components)
        print(f"    z={z:.1f}: {val:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
