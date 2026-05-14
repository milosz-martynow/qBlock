"""
Example: Using Contracted Gaussian-Type Orbitals (CGTOs) and Molecular Orbitals
================================================================================

This script demonstrates how to:
    1. Build CGTO basis functions for atoms and molecules
    2. Evaluate primitive Gaussians at arbitrary points in space
    3. Evaluate contracted basis functions
    4. Evaluate molecular orbitals from MO coefficients

Mathematical foundation:

    Primitive Gaussian:
        g_p(r) = N * (x-Rx)^lx * (y-Ry)^ly * (z-Rz)^lz * exp(-α_p * |r-R|²)

    Contracted Basis Function (CGTO):
        φ_μ(r) = Σ_p d_p * g_p(r)

    Molecular Orbital:
        ψ_i(r) = Σ_μ C_μi * φ_μ(r)
"""

from q_block.compute import Atom, Molecule
from q_block.compute.environment.io.coordinates import CartesianCoordinates
from q_block.compute.environment.io.input_data import InputData
from q_block.compute.environment.logs import setup_logging
from q_block.compute.environment.constants.numerical.pople import G631
from q_block.compute.utilities.mathematics import get_cartesian_components

logger = setup_logging(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASIS SET
# ══════════════════════════════════════════════════════════════════════════════
# Pople-style basis sets are stored in Gaussian format (.gbs files).
# The 6-31G basis is a split-valence double-zeta basis set.
logger.info("Loaded 6-31G basis set\n")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CREATE HYDROGEN ATOM WITH CGTO BASIS
# ══════════════════════════════════════════════════════════════════════════════
# An Atom can carry a basis set and coordinates.
# After calling make_contracted_gaussian_type_orbital(), the atom's CGTOs
# are built from the basis set data.

hydrogen = Atom(
    atomic_number=1,  # Hydrogen (Z=1)
    basis_set=G631,  # Attach 6-31G basis
    coordinates=CartesianCoordinates(0.0, 0.0, 0.0),  # Place at origin
)

# Build the CGTO basis functions for this atom
hydrogen.make_contracted_gaussian_type_orbital()

logger.info("=== Hydrogen atom CGTOs ===")
for i, cgto in enumerate(hydrogen.contracted_gaussian_type_orbitals):
    # l = angular momentum (0=s, 1=p, 2=d, ...)
    # n_primitives = number of primitive Gaussians in the contraction
    # n_functions = number of Cartesian basis functions (2l+1 for spherical)
    logger.info(
        f"  Shell {i}: l={cgto.l}, K={cgto.n_primitives}, "
        f"contributes {cgto.n_functions} basis function(s)"
    )
    logger.info(f"           exponents: {cgto.exponents}")
    logger.info(f"           coefficients: {cgto.contractions}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EVALUATE PRIMITIVE GAUSSIAN AT DIFFERENT POINTS
# ══════════════════════════════════════════════════════════════════════════════
# A primitive Gaussian is:
#     g(r) = N * (x-Rx)^lx * (y-Ry)^ly * (z-Rz)^lz * exp(-α|r-R|²)
#
# For s-orbitals (l=0): lx=ly=lz=0, so only the exponential part matters.
# The evaluate_primitive_gaussian() method evaluates a single primitive at a point.

cgto = hydrogen.contracted_gaussian_type_orbitals[0]  # First shell (1s core)

# Test points at different distances from the nucleus
origin = CartesianCoordinates(0.0, 0.0, 0.0)  # At the nucleus
point_x = CartesianCoordinates(0.5, 0.0, 0.0)  # 0.5 Å away
point_far = CartesianCoordinates(2.0, 0.0, 0.0)  # 2.0 Å away

logger.info("=== Primitive Gaussian evaluation" " (1s shell, primitive 0) ===")
# Arguments: (r, primitive_index, lx, ly, lz)
# For s-orbital: lx=ly=lz=0
val_origin = cgto.evaluate_primitive_gaussian(origin, 0, 0, 0, 0)
val_x = cgto.evaluate_primitive_gaussian(point_x, 0, 0, 0, 0)
val_far = cgto.evaluate_primitive_gaussian(point_far, 0, 0, 0, 0)
logger.info(f"  At origin:    {val_origin:.6f}")
logger.info(f"  At (0.5,0,0): {val_x:.6f}")
logger.info(f"  At (2.0,0,0): {val_far:.6f}")
logger.info("")
# Note: Gaussian decays exponentially with distance from the center!


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE CONTRACTED BASIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# A contracted basis function is a linear combination of primitives:
#     φ(r) = Σ_p d_p * g_p(r)
#
# The evaluate_basis_function() method computes this sum over all primitives.

logger.info("=== Contracted basis function evaluation (1s) ===")
# Arguments: (r, lx, ly, lz)
bf_origin = cgto.evaluate_basis_function(origin, 0, 0, 0)
bf_x = cgto.evaluate_basis_function(point_x, 0, 0, 0)
bf_far = cgto.evaluate_basis_function(point_far, 0, 0, 0)
logger.info(f"  At origin:    {bf_origin:.6f}")
logger.info(f"  At (0.5,0,0): {bf_x:.6f}")
logger.info(f"  At (2.0,0,0): {bf_far:.6f}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CREATE WATER MOLECULE WITH CGTOs
# ══════════════════════════════════════════════════════════════════════════════
# Molecules are built from InputData, which holds atom definitions.
# Each atom entry is: [symbol, x, y, z, basis_set]

inp = InputData()
inp.from_script(
    atom_data=[
        # Oxygen at the origin (approximately)
        ["O", 0.0000, 0.0000, 0.1173, basis],
        # Two hydrogens below oxygen
        ["H", 0.0000, 0.7572, -0.4692, basis],
        ["H", 0.0000, -0.7572, -0.4692, basis],
    ]
)

water = Molecule(input_data=inp)

# Build CGTOs for all atoms in the molecule
water.make_contracted_gaussian_type_orbital()

logger.info("=== Water molecule CGTOs ===")
logger.info(f"  Total shells: {len(water.contracted_gaussian_type_orbitals)}")
logger.info(f"  Total basis functions (n_basis): {water.n_basis}")

# Show each shell with its properties
for i, cgto in enumerate(water.contracted_gaussian_type_orbitals):
    # Shell label: s, p, d, f, g, h, i, k, l, m, ...
    shell_label = "spdfghiklm"[cgto.l] if cgto.l < 10 else f"l{cgto.l}"
    logger.info(
        f"  Shell {i:2d}: atom {cgto.atom_index}, "
        f"{shell_label}-type, "
        f"K={cgto.n_primitives}, "
        f"n_functions={cgto.n_functions}"
    )
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 6. BUILD ANGULAR COMPONENT LIST FOR MOLECULAR ORBITAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
# Each shell with angular momentum l contributes (2l+1) spherical harmonics
# or (l+1)(l+2)/2 Cartesian components.
#
# For Cartesian Gaussians, the angular components are:
#   l=0 (s): (0,0,0)                              -> 1 function
#   l=1 (p): (1,0,0), (0,1,0), (0,0,1)            -> 3 functions
#   l=2 (d): (2,0,0), (1,1,0), (1,0,1), ...       -> 6 functions

# Build the full list of angular components for all basis functions
angular_components = []
for cgto in water.contracted_gaussian_type_orbitals:
    angular_components.extend(get_cartesian_components(cgto.l))

logger.info("=== Angular components for basis functions ===")
n_comp = len(angular_components)
logger.info(f"  Total: {n_comp} " f"(should match n_basis={water.n_basis})")
logger.info(f"  First 10: {angular_components[:10]}")
logger.info("")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE MOLECULAR ORBITAL
# ══════════════════════════════════════════════════════════════════════════════
# A molecular orbital is a linear combination of basis functions:
#     ψ_i(r) = Σ_μ C_μi * φ_μ(r)
#
# In real HF calculations, coefficients come from diagonalizing the Fock matrix.
# Here we use simple test coefficients.

logger.info("=== Molecular orbital evaluation ===")

# Example 1: MO that is just the first basis function (O 1s)
# This is a unit vector selecting only the first basis function
coefficients_1s = [1.0] + [0.0] * (water.n_basis - 1)

# Evaluate near the oxygen atom
r_test = CartesianCoordinates(0.0, 0.0, 0.1173)
mo_value = water.evaluate_molecular_orbital(
    r_test, coefficients_1s, angular_components
)
logger.info(f"  MO (pure O-1s) at oxygen position: " f"{mo_value:.6f}")

# Example 2: Equal mix of first two basis functions
coefficients_mix = [0.5, 0.5] + [0.0] * (water.n_basis - 2)
mo_value_mix = water.evaluate_molecular_orbital(
    r_test, coefficients_mix, angular_components
)
logger.info(f"  MO (0.5*bf1 + 0.5*bf2) at oxygen: " f"{mo_value_mix:.6f}")

# Evaluate along z-axis to see orbital shape decay
logger.info("\n  MO (pure O-1s) along z-axis:")
for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
    r = CartesianCoordinates(0.0, 0.0, z)
    val = water.evaluate_molecular_orbital(r, coefficients_1s, angular_components)
    logger.info(f"    z={z:.1f}: {val:.6f}")

logger.info("\nDone!")
