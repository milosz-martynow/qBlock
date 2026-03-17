# ---------------------------------------------------------------------------
# Reference data: Atomic ionization energies for Koopmans' theorem testing
# ---------------------------------------------------------------------------
#
# Dictionary mapping atomic number Z (1–54) to reference data for testing
# Hartree-Fock orbital energies via Koopmans' theorem:  IE ≈ −ε_HOMO.
#
# Fields per entry:
#   symbol              – Element symbol
#   config              – Ground-state electronic configuration
#   multiplicity        – Spin multiplicity 2S+1
#   n_electrons         – Total electron count (= Z for neutral atoms)
#   n_alpha, n_beta     – UHF spin-channel occupations
#   n_closed, n_open    – ROHF doubly / singly occupied spatial orbital counts
#   proposed_hartree_fock_approach
#                       – "RHF" for closed-shell, "ROHF" for open-shell atoms
#   experimental_ie_eV  – Experimental first ionization energy in eV
#   hf_ie_eV            – Approximate Koopmans' theorem IE at
#                         the numerical Hartree-Fock limit in eV
#   proposed_basis_set  – Filename of the best available Gaussian basis set
#                         (from data/basis_set/gto_gaussian_format/) for this
#                         element, chosen as the largest basis that includes it:
#                         6-311++Gss.gbs (H, Li–Ca), 6-311G.gbs (He, Ga–Kr, I),
#                         6-31G.gbs (Sc–Zn), 3-21G.gbs (Rb–Xe except I)
#   url                 – Direct reference URL for the entry's source data
#
# Sources (openly accessible):
#   Experimental IE:
#       Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2023).
#       NIST Atomic Spectra Database (ver. 5.11).
#       National Institute of Standards and Technology, Gaithersburg, MD.
#       https://physics.nist.gov/asd
#   HF theoretical IE:
#       NIST Computational Chemistry Comparison and Benchmark Database,
#       NIST Standard Reference Database Number 101, Release 22, May 2022,
#       Editor: Russell D. Johnson III.
#       https://cccbdb.nist.gov/   DOI:10.18434/T47C7Z
#
# Notes:
#   - HF IE values are approximate.  Actual computed values depend on
#     the basis set used.  The numbers below correspond to near-basis-set-
#     limit Hartree-Fock calculations.
#   - For closed-shell atoms (multiplicity == 1), proposed_hartree_fock_approach = "RHF".
#   - For open-shell atoms  (multiplicity >  1), proposed_hartree_fock_approach = "ROHF".
#   - Conversion factor: 1 Hartree = 27.211386 eV.
# ---------------------------------------------------------------------------

ATOMS_HOMO_ENERGIES: dict = {
    # ---- Period 1 --------------------------------------------------------
    1: {
        "symbol": "H", "config": "1s1", "multiplicity": 2,
        "n_electrons": 1, "n_alpha": 1, "n_beta": 0,
        "n_closed": 0, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 13.5984, "hf_ie_eV": 13.61,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1",
    },
    2: {
        "symbol": "He", "config": "1s2", "multiplicity": 1,
        "n_electrons": 2, "n_alpha": 1, "n_beta": 1,
        "n_closed": 1, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 24.5874, "hf_ie_eV": 24.98,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=He&units=1",
    },
    # ---- Period 2 --------------------------------------------------------
    3: {
        "symbol": "Li", "config": "[He] 2s1", "multiplicity": 2,
        "n_electrons": 3, "n_alpha": 2, "n_beta": 1,
        "n_closed": 1, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 5.3917, "hf_ie_eV": 5.34,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Li&units=1",
    },
    4: {
        "symbol": "Be", "config": "[He] 2s2", "multiplicity": 1,
        "n_electrons": 4, "n_alpha": 2, "n_beta": 2,
        "n_closed": 2, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 9.3227, "hf_ie_eV": 8.42,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Be&units=1",
    },
    5: {
        "symbol": "B", "config": "[He] 2s2 2p1", "multiplicity": 2,
        "n_electrons": 5, "n_alpha": 3, "n_beta": 2,
        "n_closed": 2, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 8.2980, "hf_ie_eV": 8.43,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=B&units=1",
    },
    6: {
        "symbol": "C", "config": "[He] 2s2 2p2", "multiplicity": 3,
        "n_electrons": 6, "n_alpha": 4, "n_beta": 2,
        "n_closed": 2, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 11.2603, "hf_ie_eV": 11.79,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=C&units=1",
    },
    7: {
        "symbol": "N", "config": "[He] 2s2 2p3", "multiplicity": 4,
        "n_electrons": 7, "n_alpha": 5, "n_beta": 2,
        "n_closed": 2, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 14.5341, "hf_ie_eV": 15.44,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=N&units=1",
    },
    8: {
        "symbol": "O", "config": "[He] 2s2 2p4", "multiplicity": 3,
        "n_electrons": 8, "n_alpha": 5, "n_beta": 3,
        "n_closed": 3, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 13.6181, "hf_ie_eV": 12.44,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=O&units=1",
    },
    9: {
        "symbol": "F", "config": "[He] 2s2 2p5", "multiplicity": 2,
        "n_electrons": 9, "n_alpha": 5, "n_beta": 4,
        "n_closed": 4, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 17.4228, "hf_ie_eV": 16.22,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=F&units=1",
    },
    10: {
        "symbol": "Ne", "config": "[He] 2s2 2p6", "multiplicity": 1,
        "n_electrons": 10, "n_alpha": 5, "n_beta": 5,
        "n_closed": 5, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 21.5645, "hf_ie_eV": 23.14,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ne&units=1",
    },
    # ---- Period 3 --------------------------------------------------------
    11: {
        "symbol": "Na", "config": "[Ne] 3s1", "multiplicity": 2,
        "n_electrons": 11, "n_alpha": 6, "n_beta": 5,
        "n_closed": 5, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 5.1391, "hf_ie_eV": 5.02,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Na&units=1",
    },
    12: {
        "symbol": "Mg", "config": "[Ne] 3s2", "multiplicity": 1,
        "n_electrons": 12, "n_alpha": 6, "n_beta": 6,
        "n_closed": 6, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 7.6462, "hf_ie_eV": 6.89,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mg&units=1",
    },
    13: {
        "symbol": "Al", "config": "[Ne] 3s2 3p1", "multiplicity": 2,
        "n_electrons": 13, "n_alpha": 7, "n_beta": 6,
        "n_closed": 6, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 5.9858, "hf_ie_eV": 5.71,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Al&units=1",
    },
    14: {
        "symbol": "Si", "config": "[Ne] 3s2 3p2", "multiplicity": 3,
        "n_electrons": 14, "n_alpha": 8, "n_beta": 6,
        "n_closed": 6, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 8.1517, "hf_ie_eV": 8.14,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Si&units=1",
    },
    15: {
        "symbol": "P", "config": "[Ne] 3s2 3p3", "multiplicity": 4,
        "n_electrons": 15, "n_alpha": 9, "n_beta": 6,
        "n_closed": 6, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 10.4867, "hf_ie_eV": 10.67,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=P&units=1",
    },
    16: {
        "symbol": "S", "config": "[Ne] 3s2 3p4", "multiplicity": 3,
        "n_electrons": 16, "n_alpha": 9, "n_beta": 7,
        "n_closed": 7, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 10.3600, "hf_ie_eV": 9.63,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=S&units=1",
    },
    17: {
        "symbol": "Cl", "config": "[Ne] 3s2 3p5", "multiplicity": 2,
        "n_electrons": 17, "n_alpha": 9, "n_beta": 8,
        "n_closed": 8, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 12.9676, "hf_ie_eV": 12.42,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cl&units=1",
    },
    18: {
        "symbol": "Ar", "config": "[Ne] 3s2 3p6", "multiplicity": 1,
        "n_electrons": 18, "n_alpha": 9, "n_beta": 9,
        "n_closed": 9, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 15.7596, "hf_ie_eV": 16.08,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ar&units=1",
    },
    # ---- Period 4 --------------------------------------------------------
    19: {
        "symbol": "K", "config": "[Ar] 4s1", "multiplicity": 2,
        "n_electrons": 19, "n_alpha": 10, "n_beta": 9,
        "n_closed": 9, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 4.3407, "hf_ie_eV": 4.19,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=K&units=1",
    },
    20: {
        "symbol": "Ca", "config": "[Ar] 4s2", "multiplicity": 1,
        "n_electrons": 20, "n_alpha": 10, "n_beta": 10,
        "n_closed": 10, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 6.1132, "hf_ie_eV": 5.49,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ca&units=1",
    },
    21: {
        "symbol": "Sc", "config": "[Ar] 3d1 4s2", "multiplicity": 2,
        "n_electrons": 21, "n_alpha": 11, "n_beta": 10,
        "n_closed": 10, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.5615, "hf_ie_eV": 5.67,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sc&units=1",
    },
    22: {
        "symbol": "Ti", "config": "[Ar] 3d2 4s2", "multiplicity": 3,
        "n_electrons": 22, "n_alpha": 12, "n_beta": 10,
        "n_closed": 10, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.8281, "hf_ie_eV": 5.98,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ti&units=1",
    },
    23: {
        "symbol": "V", "config": "[Ar] 3d3 4s2", "multiplicity": 4,
        "n_electrons": 23, "n_alpha": 13, "n_beta": 10,
        "n_closed": 10, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.7462, "hf_ie_eV": 5.92,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=V&units=1",
    },
    24: {
        "symbol": "Cr", "config": "[Ar] 3d5 4s1", "multiplicity": 7,
        "n_electrons": 24, "n_alpha": 15, "n_beta": 9,
        "n_closed": 9, "n_open": 6,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.7665, "hf_ie_eV": 5.79,
        "proposed_basis_set": "6-31G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cr&units=1",
    },
    25: {
        "symbol": "Mn", "config": "[Ar] 3d5 4s2", "multiplicity": 6,
        "n_electrons": 25, "n_alpha": 15, "n_beta": 10,
        "n_closed": 10, "n_open": 5,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.4340, "hf_ie_eV": 6.66,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mn&units=1",
    },
    26: {
        "symbol": "Fe", "config": "[Ar] 3d6 4s2", "multiplicity": 5,
        "n_electrons": 26, "n_alpha": 15, "n_beta": 11,
        "n_closed": 11, "n_open": 4,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.9024, "hf_ie_eV": 6.65,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Fe&units=1",
    },
    27: {
        "symbol": "Co", "config": "[Ar] 3d7 4s2", "multiplicity": 4,
        "n_electrons": 27, "n_alpha": 15, "n_beta": 12,
        "n_closed": 12, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.8810, "hf_ie_eV": 6.50,
        "proposed_basis_set": "6-31G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Co&units=1",
    },
    28: {
        "symbol": "Ni", "config": "[Ar] 3d8 4s2", "multiplicity": 3,
        "n_electrons": 28, "n_alpha": 15, "n_beta": 13,
        "n_closed": 13, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.6398, "hf_ie_eV": 6.42,
        "proposed_basis_set": "6-31G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ni&units=1",
    },
    29: {
        "symbol": "Cu", "config": "[Ar] 3d10 4s1", "multiplicity": 2,
        "n_electrons": 29, "n_alpha": 15, "n_beta": 14,
        "n_closed": 14, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.7264, "hf_ie_eV": 6.49,
        "proposed_basis_set": "6-31G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cu&units=1",
    },
    30: {
        "symbol": "Zn", "config": "[Ar] 3d10 4s2", "multiplicity": 1,
        "n_electrons": 30, "n_alpha": 15, "n_beta": 15,
        "n_closed": 15, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 9.3942, "hf_ie_eV": 8.61,
        "proposed_basis_set": "6-31G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Zn&units=1",
    },
    31: {
        "symbol": "Ga", "config": "[Ar] 3d10 4s2 4p1", "multiplicity": 2,
        "n_electrons": 31, "n_alpha": 16, "n_beta": 15,
        "n_closed": 15, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 5.9993, "hf_ie_eV": 5.59,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ga&units=1",
    },
    32: {
        "symbol": "Ge", "config": "[Ar] 3d10 4s2 4p2", "multiplicity": 3,
        "n_electrons": 32, "n_alpha": 17, "n_beta": 15,
        "n_closed": 15, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.8994, "hf_ie_eV": 7.76,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ge&units=1",
    },
    33: {
        "symbol": "As", "config": "[Ar] 3d10 4s2 4p3", "multiplicity": 4,
        "n_electrons": 33, "n_alpha": 18, "n_beta": 15,
        "n_closed": 15, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 9.7886, "hf_ie_eV": 9.95,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=As&units=1",
    },
    34: {
        "symbol": "Se", "config": "[Ar] 3d10 4s2 4p4", "multiplicity": 3,
        "n_electrons": 34, "n_alpha": 18, "n_beta": 16,
        "n_closed": 16, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 9.7524, "hf_ie_eV": 8.99,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Se&units=1",
    },
    35: {
        "symbol": "Br", "config": "[Ar] 3d10 4s2 4p5", "multiplicity": 2,
        "n_electrons": 35, "n_alpha": 18, "n_beta": 17,
        "n_closed": 17, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 11.8138, "hf_ie_eV": 11.23,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Br&units=1",
    },
    36: {
        "symbol": "Kr", "config": "[Ar] 3d10 4s2 4p6", "multiplicity": 1,
        "n_electrons": 36, "n_alpha": 18, "n_beta": 18,
        "n_closed": 18, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 13.9996, "hf_ie_eV": 14.26,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Kr&units=1",
    },
    # ---- Period 5 --------------------------------------------------------
    37: {
        "symbol": "Rb", "config": "[Kr] 5s1", "multiplicity": 2,
        "n_electrons": 37, "n_alpha": 19, "n_beta": 18,
        "n_closed": 18, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 4.1771, "hf_ie_eV": 3.98,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Rb&units=1",
    },
    38: {
        "symbol": "Sr", "config": "[Kr] 5s2", "multiplicity": 1,
        "n_electrons": 38, "n_alpha": 19, "n_beta": 19,
        "n_closed": 19, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 5.6949, "hf_ie_eV": 5.03,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sr&units=1",
    },
    39: {
        "symbol": "Y", "config": "[Kr] 4d1 5s2", "multiplicity": 2,
        "n_electrons": 39, "n_alpha": 20, "n_beta": 19,
        "n_closed": 19, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.2173, "hf_ie_eV": 5.39,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Y&units=1",
    },
    40: {
        "symbol": "Zr", "config": "[Kr] 4d2 5s2", "multiplicity": 3,
        "n_electrons": 40, "n_alpha": 21, "n_beta": 19,
        "n_closed": 19, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.6339, "hf_ie_eV": 5.70,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Zr&units=1",
    },
    41: {
        "symbol": "Nb", "config": "[Kr] 4d4 5s1", "multiplicity": 6,
        "n_electrons": 41, "n_alpha": 23, "n_beta": 18,
        "n_closed": 18, "n_open": 5,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 6.7589, "hf_ie_eV": 5.82,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Nb&units=1",
    },
    42: {
        "symbol": "Mo", "config": "[Kr] 4d5 5s1", "multiplicity": 7,
        "n_electrons": 42, "n_alpha": 24, "n_beta": 18,
        "n_closed": 18, "n_open": 6,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.0924, "hf_ie_eV": 6.02,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mo&units=1",
    },
    43: {
        "symbol": "Tc", "config": "[Kr] 4d5 5s2", "multiplicity": 6,
        "n_electrons": 43, "n_alpha": 24, "n_beta": 19,
        "n_closed": 19, "n_open": 5,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.1190, "hf_ie_eV": 6.37,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Tc&units=1",
    },
    44: {
        "symbol": "Ru", "config": "[Kr] 4d7 5s1", "multiplicity": 5,
        "n_electrons": 44, "n_alpha": 24, "n_beta": 20,
        "n_closed": 20, "n_open": 4,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.3605, "hf_ie_eV": 6.33,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ru&units=1",
    },
    45: {
        "symbol": "Rh", "config": "[Kr] 4d8 5s1", "multiplicity": 4,
        "n_electrons": 45, "n_alpha": 24, "n_beta": 21,
        "n_closed": 21, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.4589, "hf_ie_eV": 6.30,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Rh&units=1",
    },
    46: {
        "symbol": "Pd", "config": "[Kr] 4d10", "multiplicity": 1,
        "n_electrons": 46, "n_alpha": 23, "n_beta": 23,
        "n_closed": 23, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 8.3369, "hf_ie_eV": 7.40,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Pd&units=1",
    },
    47: {
        "symbol": "Ag", "config": "[Kr] 4d10 5s1", "multiplicity": 2,
        "n_electrons": 47, "n_alpha": 24, "n_beta": 23,
        "n_closed": 23, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.5762, "hf_ie_eV": 6.25,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ag&units=1",
    },
    48: {
        "symbol": "Cd", "config": "[Kr] 4d10 5s2", "multiplicity": 1,
        "n_electrons": 48, "n_alpha": 24, "n_beta": 24,
        "n_closed": 24, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 8.9938, "hf_ie_eV": 8.15,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cd&units=1",
    },
    49: {
        "symbol": "In", "config": "[Kr] 4d10 5s2 5p1", "multiplicity": 2,
        "n_electrons": 49, "n_alpha": 25, "n_beta": 24,
        "n_closed": 24, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 5.7864, "hf_ie_eV": 5.27,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=In&units=1",
    },
    50: {
        "symbol": "Sn", "config": "[Kr] 4d10 5s2 5p2", "multiplicity": 3,
        "n_electrons": 50, "n_alpha": 26, "n_beta": 24,
        "n_closed": 24, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 7.3439, "hf_ie_eV": 7.16,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sn&units=1",
    },
    51: {
        "symbol": "Sb", "config": "[Kr] 4d10 5s2 5p3", "multiplicity": 4,
        "n_electrons": 51, "n_alpha": 27, "n_beta": 24,
        "n_closed": 24, "n_open": 3,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 8.6084, "hf_ie_eV": 8.76,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sb&units=1",
    },
    52: {
        "symbol": "Te", "config": "[Kr] 4d10 5s2 5p4", "multiplicity": 3,
        "n_electrons": 52, "n_alpha": 27, "n_beta": 25,
        "n_closed": 25, "n_open": 2,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 9.0096, "hf_ie_eV": 8.24,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Te&units=1",
    },
    53: {
        "symbol": "I", "config": "[Kr] 4d10 5s2 5p5", "multiplicity": 2,
        "n_electrons": 53, "n_alpha": 27, "n_beta": 26,
        "n_closed": 26, "n_open": 1,
        "proposed_hartree_fock_approach": "ROHF",
        "experimental_ie_eV": 10.4513, "hf_ie_eV": 10.07,
        "proposed_basis_set": "6-311G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=I&units=1",
    },
    54: {
        "symbol": "Xe", "config": "[Kr] 4d10 5s2 5p6", "multiplicity": 1,
        "n_electrons": 54, "n_alpha": 27, "n_beta": 27,
        "n_closed": 27, "n_open": 0,
        "proposed_hartree_fock_approach": "RHF",
        "experimental_ie_eV": 12.1298, "hf_ie_eV": 12.44,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Xe&units=1",
    },
}

# ---------------------------------------------------------------------------
# Reference data: Molecular HOMO energies for Koopmans' theorem testing
# ---------------------------------------------------------------------------
#
# Dictionary mapping molecule identifier to reference data for validating
# Hartree-Fock orbital energies via Koopmans' theorem:  IE ≈ −ε_HOMO.
#
# Entries are grouped by the target HF method:
#   RHF  – restricted HF for closed-shell singlets  (multiplicity == 1)
#   ROHF – restricted open-shell HF                 (multiplicity >  1)
#   UHF  – unrestricted HF for open-shell systems   (multiplicity >  1)
#
# Fields per entry:
#   formula             – Molecular formula with Unicode superscripts
#   name                – Common / popular name (empty string if none)
#   technical_name      – SMILES (Simplified Molecular Input Line Entry System)
#   geometry            – Equilibrium geometry as a list of
#                         {"symbol": str, "x": float, "y": float, "z": float}
#                         dicts; Cartesian coordinates in Ångström
#                         (experimental Re values, z-axis along bond /
#                         principal axis).  Source: NIST CCCBDB.
#   proposed_hartree_fock_approach
#                       – Target HF method: "RHF", "ROHF", or "UHF"
#   multiplicity        – Spin multiplicity 2S+1
#   n_electrons         – Total electron count
#   n_alpha, n_beta     – Alpha / beta spin-orbital occupations
#   n_closed, n_open    – ROHF doubly / singly occupied spatial orbital counts
#   experimental_ie_eV  – Experimental vertical first ionization energy in eV
#   hf_ie_eV            – Approximate Koopmans' theorem IE at the near-basis-
#                         set-limit for the given method, in eV
#   proposed_basis_set  – Filename of the best available Gaussian basis set
#                         (from data/basis_set/gto_gaussian_format/) for all
#                         atoms in the molecule.  Uses the largest basis whose
#                         element coverage includes every constituent atom.
#   url                 – Direct reference URL for the entry's source data
#
# Sources:
#   Experimental IE:
#       Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2023).
#       NIST Atomic Spectra Database (ver. 5.11).
#       https://physics.nist.gov/asd
#   Geometries:
#       NIST CCCBDB experimental geometries.
#       https://cccbdb.nist.gov/
#   HF reference IE:
#       NIST CCCBDB, Release 22, May 2022.  DOI:10.18434/T47C7Z
#
# Notes:
#   - For UHF the reported hf_ie_eV corresponds to −ε_α_HOMO (alpha-spin HOMO).
#   - Conversion factor: 1 Hartree = 27.211386 eV.
# ---------------------------------------------------------------------------

MOLECULES_HOMO_ENERGIES: dict = {
    # ---- RHF molecules (closed-shell, multiplicity == 1) -----------------

    # Hydrogen molecule: simplest two-electron system; HOMO = 1σg
    "H2": {
        "formula": "H₂",
        "name": "hydrogen",
        "technical_name": "[HH]",
        "geometry": [
            {"symbol": "H", "x":  0.0000, "y": 0.0000, "z":  0.3707},
            {"symbol": "H", "x":  0.0000, "y": 0.0000, "z": -0.3707},
        ],
        "proposed_hartree_fock_approach": "RHF",
        "multiplicity": 1,
        "n_electrons": 2, "n_alpha": 1, "n_beta": 1,
        "n_closed": 1, "n_open": 0,
        "experimental_ie_eV": 15.4259, "hf_ie_eV": 16.17,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Water: HOMO = 1b₁ (lone pair on O perpendicular to molecular plane)
    "H2O": {
        "formula": "H₂O",
        "name": "water",
        "technical_name": "O",
        "geometry": [
            {"symbol": "O", "x":  0.0000, "y": 0.0000, "z":  0.0000},
            {"symbol": "H", "x":  0.7572, "y": 0.0000, "z":  0.5860},
            {"symbol": "H", "x": -0.7572, "y": 0.0000, "z":  0.5860},
        ],
        "proposed_hartree_fock_approach": "RHF",
        "multiplicity": 1,
        "n_electrons": 10, "n_alpha": 5, "n_beta": 5,
        "n_closed": 5, "n_open": 0,
        "experimental_ie_eV": 12.621, "hf_ie_eV": 13.79,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Dinitrogen: HOMO = 3σg (note: 3σg lies above 1πu in HF ordering)
    "N2": {
        "formula": "N₂",
        "name": "nitrogen",
        "technical_name": "N#N",
        "geometry": [
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z":  0.5489},
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5489},
        ],
        "proposed_hartree_fock_approach": "RHF",
        "multiplicity": 1,
        "n_electrons": 14, "n_alpha": 7, "n_beta": 7,
        "n_closed": 7, "n_open": 0,
        "experimental_ie_eV": 15.5808, "hf_ie_eV": 16.71,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Carbon monoxide: HOMO = 5σ (weakly bonding, primarily C lone pair)
    "CO": {
        "formula": "CO",
        "name": "carbon monoxide",
        "technical_name": "[C-]#[O+]",
        "geometry": [
            {"symbol": "C", "x": 0.0000, "y": 0.0000, "z": -0.5641},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z":  0.5641},
        ],
        "proposed_hartree_fock_approach": "RHF",
        "multiplicity": 1,
        "n_electrons": 14, "n_alpha": 7, "n_beta": 7,
        "n_closed": 7, "n_open": 0,
        "experimental_ie_eV": 14.014, "hf_ie_eV": 15.05,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },

    # ---- ROHF molecules (open-shell, fixed multiplicity) -----------------

    # Dioxygen: triplet ground state; HOMO = 1πg (degenerate, singly occupied)
    "O2": {
        "formula": "O₂",
        "name": "oxygen",
        "technical_name": "[O][O]",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z":  0.6038},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": -0.6038},
        ],
        "proposed_hartree_fock_approach": "ROHF",
        "multiplicity": 3,
        "n_electrons": 16, "n_alpha": 9, "n_beta": 7,
        "n_closed": 7, "n_open": 2,
        "experimental_ie_eV": 12.0697, "hf_ie_eV": 15.84,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Nitric oxide: doublet; HOMO = 2π (one electron in antibonding π*)
    "NO": {
        "formula": "NO",
        "name": "nitric oxide",
        "technical_name": "[N]=O",
        "geometry": [
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5754},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z":  0.5754},
        ],
        "proposed_hartree_fock_approach": "ROHF",
        "multiplicity": 2,
        "n_electrons": 15, "n_alpha": 8, "n_beta": 7,
        "n_closed": 7, "n_open": 1,
        "experimental_ie_eV": 9.2643, "hf_ie_eV": 11.51,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Hydroxyl radical: doublet; HOMO = 1π (degenerate, one singly occupied)
    "OH": {
        "formula": "OH",
        "name": "hydroxyl",
        "technical_name": "[OH]",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": -0.4849},
            {"symbol": "H", "x": 0.0000, "y": 0.0000, "z":  0.4849},
        ],
        "proposed_hartree_fock_approach": "ROHF",
        "multiplicity": 2,
        "n_electrons": 9, "n_alpha": 5, "n_beta": 4,
        "n_closed": 4, "n_open": 1,
        "experimental_ie_eV": 13.017, "hf_ie_eV": 13.02,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },

    # ---- UHF molecules (open-shell, spin-unrestricted) -------------------
    # hf_ie_eV = −ε_α_HOMO (alpha-spin HOMO energy).  UHF introduces spin
    # contamination; <S²> may deviate from the exact S(S+1) value.

    # Dioxygen (UHF): same geometry as ROHF reference above
    "O2": {
        "formula": "O₂",
        "name": "oxygen",
        "technical_name": "[O][O]",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z":  0.6038},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": -0.6038},
        ],
        "proposed_hartree_fock_approach": "UHF",
        "multiplicity": 3,
        "n_electrons": 16, "n_alpha": 9, "n_beta": 7,
        "n_closed": 7, "n_open": 2,
        "experimental_ie_eV": 12.0697, "hf_ie_eV": 15.87,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Nitric oxide (UHF)
    "NO": {
        "formula": "NO",
        "name": "nitric oxide",
        "technical_name": "[N]=O",
        "geometry": [
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5754},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z":  0.5754},
        ],
        "proposed_hartree_fock_approach": "UHF",
        "multiplicity": 2,
        "n_electrons": 15, "n_alpha": 8, "n_beta": 7,
        "n_closed": 7, "n_open": 1,
        "experimental_ie_eV": 9.2643, "hf_ie_eV": 11.54,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Methyl radical: doublet D₃h; HOMO = 2a₂'' (singly occupied, p_z on C)
    "CH3": {
        "formula": "CH₃",
        "name": "methyl radical",
        "technical_name": "[CH3]",
        "geometry": [
            {"symbol": "C", "x":  0.0000, "y":  0.0000, "z": 0.0000},
            {"symbol": "H", "x":  1.0790, "y":  0.0000, "z": 0.0000},
            {"symbol": "H", "x": -0.5395, "y":  0.9344, "z": 0.0000},
            {"symbol": "H", "x": -0.5395, "y": -0.9344, "z": 0.0000},
        ],
        "proposed_hartree_fock_approach": "UHF",
        "multiplicity": 2,
        "n_electrons": 9, "n_alpha": 5, "n_beta": 4,
        "n_closed": 4, "n_open": 1,
        "experimental_ie_eV": 9.840, "hf_ie_eV": 11.40,
        "proposed_basis_set": "6-311++Gss.gbs",
        "url": "https://cccbdb.nist.gov/",
    },

    # ---- Molecules containing elements with empirical electron-configuration exceptions ------
    #
    # Chromium monoxide: X⁵Π ground state; Cr is [Ar]3d⁵4s¹ (empirical exception) coupled to O
    # giving 4 singly occupied 3d-based MOs.  Good ROHF/UHF test because HF captures the
    # qualitative spin structure but misses 3d dynamical correlation, so the Koopmans IE
    # overestimates experiment by ~25 % (larger error than for first-row open-shell molecules).
    #
    # Geometry source: Huber & Herzberg (1979), via NIST Webbook constants of diatomic molecules.
    #   Re = 1.615 Å  (Be = 0.5410 cm⁻¹, X⁵Π).
    # Experimental IE source: Dyke, Gravenor, Lewis, Morris,
    #   J. Chem. Soc. Faraday Trans. 2, 1983, 79, 2083.
    #   Vertical IE = 8.16 ± 0.01 eV (PE spectroscopy; adiabatic IE = 7.85 ± 0.02 eV).
    # HF reference: approximate near-basis-set-limit ROHF/UHF Koopmans value from published
    #   benchmark calculations (NOT from NIST CCCBDB, which does not cover this system).
    #   Value is strongly basis-set-dependent; treat as ±0.5 eV.

    # Chromium monoxide (ROHF): highest open-shell orbital is a 3d-based π/δ SOMO
    "CrO": {
        "formula": "CrO",
        "name": "chromium monoxide",
        "technical_name": "[Cr]=O",
        "geometry": [
            {"symbol": "Cr", "x": 0.0000, "y": 0.0000, "z": -0.8075},
            {"symbol": "O",  "x": 0.0000, "y": 0.0000, "z":  0.8075},
        ],
        "proposed_hartree_fock_approach": "ROHF",
        "multiplicity": 5,
        "n_electrons": 32, "n_alpha": 18, "n_beta": 14,
        "n_closed": 14, "n_open": 4,
        "experimental_ie_eV": 8.16, "hf_ie_eV": 10.19,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
    # Chromium monoxide (UHF): same geometry; −ε_α_HOMO differs slightly from ROHF
    # due to spin-polarisation of the alpha and beta orbital spaces.
    "CrO": {
        "formula": "CrO",
        "name": "chromium monoxide",
        "technical_name": "[Cr]=O",
        "geometry": [
            {"symbol": "Cr", "x": 0.0000, "y": 0.0000, "z": -0.8075},
            {"symbol": "O",  "x": 0.0000, "y": 0.0000, "z":  0.8075},
        ],
        "proposed_hartree_fock_approach": "UHF",
        "multiplicity": 5,
        "n_electrons": 32, "n_alpha": 18, "n_beta": 14,
        "n_closed": 14, "n_open": 4,
        "experimental_ie_eV": 8.16, "hf_ie_eV": 10.27,
        "proposed_basis_set": "3-21G.gbs",
        "url": "https://cccbdb.nist.gov/",
    },
}
