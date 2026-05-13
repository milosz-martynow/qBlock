# TO DO: FIX TESTS THAT ARE FAILING. PROBABLY NEED TO USE BIGGER BASIS SETS IN SOME OF EXAMPLES.
# TO DO: UKS is not for 1> multiplicity > 1 but for 0.5 multiplicity odd-electron systems.  Need to check that all proposed_approach values are correct.

# ---------------------------------------------------------------------------
# Running tests for specific records
# ---------------------------------------------------------------------------
#
# To test a specific atom or molecule, use pytest's -k flag with the test ID:
#
# ATOMS: Test IDs are formatted as Z{atomic_number}_{symbol}
#   Examples:
#     pytest .\tests\validation\test_uhf.py -k "Z9_F"       # Fluorine
#     pytest .\tests\validation\test_rhf.py -k "Z2_He"      # Helium
#     pytest .\tests\validation\ -k "Z17_Cl"                # Chlorine (all test files)
#     pytest .\tests\validation\ -k "Z9_F or Z17_Cl"        # Multiple atoms
#
# MOLECULES: Test IDs are the dictionary keys (e.g., "H2O", "CH4")
#   Examples:
#     pytest .\tests\validation\test_rhf.py -k "H2O"        # Water
#     pytest .\tests\validation\test_uhf.py -k "NO"         # Nitric oxide
#
# To see all available test IDs without running:
#     pytest .\tests\validation\test_rhf.py --collect-only
#
#
# ATOMS_HOMO_ENERGIES  (int key = Z) and MOLECULES_HOMO_ENERGIES  (str key)
# store reference ionization energies (IE) used to validate Hartree-Fock and
# DFT orbital energies via Koopmans' theorem:  IE ≈ −ε_HOMO.
#
# Shared entry schema:
#   "multiplicity":       int         – spin multiplicity 2S+1
#   "n_electrons":        int         – total electron count
#   "n_alpha", "n_beta":  int         – alpha / beta spin-orbital occupations
#   "n_closed", "n_open": int         – ROHF doubly / singly occupied spatial orbitals
#   "proposed_approach":  Dict[str, Dict] – applicable methods mapped to their config.
#                                           Keys are any subset of
#                                           ["RHF", "ROHF", "UHF", "RKS", "UKS", "ROKS"].
#                                           Each value dict contains:
#                                             "proposed_basis_set": str  – Gaussian basis set path
#                                             "max_iterations":     int  – max SCF iterations
#                                             "functional": List[str]    – (DFT only) any subset of
#                                                           ["B3LYP", "PBE", "SVWN"]; absent for HF
#   "energies": {
#       "experiment": List[{"value": float, "reference": str,
#                           "note": str, "level": int}],
#       "hf":         List[{"value": float, "reference": str,
#                           "note": str, "level": int}],
#       "dft":        List[{"value": float, "reference": str,
#                           "note": str, "level": int}],
#   }
#
# energies sub-fields:
#   "value"     – IE in eV
#   "reference" – source URL
#   "note"      – free-text annotation ("" if none)
#   "level"     – orbital level relative to HOMO: 0 = HOMO, −n = n below,
#                 +n = n above
#
# energies["experiment"] – experimental IE (NIST ASD for atoms;
#                          NIST WebBook for molecules)
# energies["hf"]         – HF Koopmans' IE (−ε_HOMO); for UHF: −ε_α_HOMO
# energies["dft"]        – B3LYP/3-21G adiabatic IE (delta-SCF, NIST CCCBDB);
#                          empty list when no CCCBDB value is available
#
# Sources:
#   Atoms – experimental IE:
#       NIST Atomic Spectra Database (ASD), ver. 5.11.
#       https://physics.nist.gov/asd
#   Atoms – HF IE:
#       Krishnamohan G P et al. (2017), World J. Chem. Educ. 5(3), 112–119.
#       DOI: 10.12691/wjce-5-3-6
#   Molecules – experimental IE:
#       NIST Chemistry WebBook, SRD 69.  https://webbook.nist.gov/
#   Molecules – geometries:
#       NIST CCCBDB experimental geometries.  https://cccbdb.nist.gov/
#   DFT IE (atoms and molecules):
#       NIST CCCBDB, SRD 101, Release 22 (May 2022).
#       https://cccbdb.nist.gov/   DOI:10.18434/T47C7Z
#   HF IE (molecules):
#       NIST CCCBDB (same reference as above).
#
# Notes:
#   - HF IE values are basis-set-dependent approximations.
#   - DFT IE values are adiabatic (delta-SCF), not Koopmans'.
#   - Conversion factor: 1 Hartree = 27.211386 eV.
# ---------------------------------------------------------------------------

ATOMS: dict = {
    # ---- Period 1 --------------------------------------------------------
    1: {
        "symbol": "H",
        "config": "1s1",
        "multiplicity": 2,
        "n_electrons": 1,
        "n_alpha": 1,
        "n_beta": 0,
        "n_closed": 0,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 13.5984, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 13.61, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 13.533, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=12385-13-6', "note": "", "level": 0},
            ],
        },
    },
    # There is a problem with DFT for He atom. Need to add more functionals, and basis sets readers.
    2: {
        "symbol": "He",
        "config": "1s2",
        "multiplicity": 1,
        "n_electrons": 2,
        "n_alpha": 1,
        "n_beta": 1,
        "n_closed": 1,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
        },
        "energies": {
            "experiment": [
                {"value": 24.5874, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=He&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 24.98, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    # ---- Period 2 --------------------------------------------------------
    3: {
        "symbol": "Li",
        "config": "[He] 2s1",
        "multiplicity": 2,
        "n_electrons": 3,
        "n_alpha": 2,
        "n_beta": 1,
        "n_closed": 1,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 5.3917, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Li&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.34, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 5.587, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7439-93-2', "note": "", "level": 0},
            ],
        },
    },
    4: {
        "symbol": "Be",
        "config": "[He] 2s2",
        "multiplicity": 1,
        "n_electrons": 4,
        "n_alpha": 2,
        "n_beta": 2,
        "n_closed": 2,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 9.3227, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Be&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 8.42, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 9.179, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-41-7', "note": "", "level": 0},
            ],
        },
    },
    5: {
        "symbol": "B",
        "config": "[He] 2s2 2p1",
        "multiplicity": 2,
        "n_electrons": 5,
        "n_alpha": 3,
        "n_beta": 2,
        "n_closed": 2,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 8.298, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=B&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 8.43, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 8.74, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-42-8', "note": "", "level": 0},
            ],
        },
    },
    6: {
        "symbol": "C",
        "config": "[He] 2s2 2p2",
        "multiplicity": 3,
        "n_electrons": 6,
        "n_alpha": 4,
        "n_beta": 2,
        "n_closed": 2,
        "n_open": 2,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 11.2603, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=C&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 11.79, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 11.509, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-44-0', "note": "", "level": 0},
            ],
        },
    },
    7: {
        "symbol": "N",
        "config": "[He] 2s2 2p3",
        "multiplicity": 4,
        "n_electrons": 7,
        "n_alpha": 5,
        "n_beta": 2,
        "n_closed": 2,
        "n_open": 3,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 14.5341, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=N&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 15.44, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 14.507, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=17778-88-0', "note": "", "level": 0},
            ],
        },
    },
    # Oxygen is an examle where HF methods like "ROHF", "UHF" are failing.
    8: {
        "symbol": "O",
        "config": "[He] 2s2 2p4",
        "multiplicity": 3,
        "n_electrons": 8,
        "n_alpha": 5,
        "n_beta": 3,
        "n_closed": 3,
        "n_open": 2,
        "proposed_approach": {
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 13.6181, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=O&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 12.44, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 13.622, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=17778-80-2', "note": "", "level": 0},
            ],
        },
    },
    9: {
        "symbol": "F",
        "config": "[He] 2s2 2p5",
        "multiplicity": 2,
        "n_electrons": 9,
        "n_alpha": 5,
        "n_beta": 4,
        "n_closed": 4,
        "n_open": 1,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 17.4228, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=F&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 16.22, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 16.935, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=14762-94-8', "note": "", "level": 0},
            ],
        },
    },
    10: {
        "symbol": "Ne",
        "config": "[He] 2s2 2p6",
        "multiplicity": 1,
        "n_electrons": 10,
        "n_alpha": 5,
        "n_beta": 5,
        "n_closed": 5,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311++G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 21.5645, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ne&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 23.14, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 20.552, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-01-9', "note": "", "level": 0},
            ],
        },
    },
    # ---- Period 3 --------------------------------------------------------
    11: {
        "symbol": "Na",
        "config": "[Ne] 3s1",
        "multiplicity": 2,
        "n_electrons": 11,
        "n_alpha": 6,
        "n_beta": 5,
        "n_closed": 5,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 5.1391, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Na&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.02, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 5.359, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-23-5', "note": "", "level": 0},
            ],
        },
    },
    12: {
        "symbol": "Mg",
        "config": "[Ne] 3s2",
        "multiplicity": 1,
        "n_electrons": 12,
        "n_alpha": 6,
        "n_beta": 6,
        "n_closed": 6,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.6462, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mg&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.89, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 7.745, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7439-95-4', "note": "", "level": 0},
            ],
        },
    },
    13: {
        "symbol": "Al",
        "config": "[Ne] 3s2 3p1",
        "multiplicity": 2,
        "n_electrons": 13,
        "n_alpha": 7,
        "n_beta": 6,
        "n_closed": 6,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 5.9858, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Al&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.71, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 6.022, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7429-90-5', "note": "", "level": 0},
            ],
        },
    },
    14: {
        "symbol": "Si",
        "config": "[Ne] 3s2 3p2",
        "multiplicity": 3,
        "n_electrons": 14,
        "n_alpha": 8,
        "n_beta": 6,
        "n_closed": 6,
        "n_open": 2,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 8.1517, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Si&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 8.14, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 8.141, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-21-3', "note": "", "level": 0},
            ],
        },
    },
    15: {
        "symbol": "P",
        "config": "[Ne] 3s2 3p3",
        "multiplicity": 4,
        "n_electrons": 15,
        "n_alpha": 9,
        "n_beta": 6,
        "n_closed": 6,
        "n_open": 3,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 10.4867, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=P&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 10.67, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 10.432, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7723-14-0', "note": "", "level": 0},
            ],
        },
    },
    16: {
        "symbol": "S",
        "config": "[Ne] 3s2 3p4",
        "multiplicity": 3,
        "n_electrons": 16,
        "n_alpha": 9,
        "n_beta": 7,
        "n_closed": 7,
        "n_open": 2,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 10.36, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=S&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 9.63, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 10.483, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7704-34-9', "note": "", "level": 0},
            ],
        },
    },
    17: {
        "symbol": "Cl",
        "config": "[Ne] 3s2 3p5",
        "multiplicity": 2,
        "n_electrons": 17,
        "n_alpha": 9,
        "n_beta": 8,
        "n_closed": 8,
        "n_open": 1,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 12.9676, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cl&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 12.42, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 13.059, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=22537-15-1', "note": "", "level": 0},
            ],
        },
    },
    18: {
        "symbol": "Ar",
        "config": "[Ne] 3s2 3p6",
        "multiplicity": 1,
        "n_electrons": 18,
        "n_alpha": 9,
        "n_beta": 9,
        "n_closed": 9,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 15.7596, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ar&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 16.08, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 15.856, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-37-1', "note": "", "level": 0},
            ],
        },
    },
    # ---- Period 4 --------------------------------------------------------
    19: {
        "symbol": "K",
        "config": "[Ar] 4s1",
        "multiplicity": 2,
        "n_electrons": 19,
        "n_alpha": 10,
        "n_beta": 9,
        "n_closed": 9,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 4.3407, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=K&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 4.19, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 4.452, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-09-7', "note": "", "level": 0},
            ],
        },
    },
    20: {
        "symbol": "Ca",
        "config": "[Ar] 4s2",
        "multiplicity": 1,
        "n_electrons": 20,
        "n_alpha": 10,
        "n_beta": 10,
        "n_closed": 10,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 300,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 300,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 6.1132, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ca&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.49, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    21: {
        "symbol": "Sc",
        "config": "[Ar] 3d1 4s2",
        "multiplicity": 2,
        "n_electrons": 21,
        "n_alpha": 11,
        "n_beta": 10,
        "n_closed": 10,
        "n_open": 1,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 6.5615, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sc&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.67, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 6.447, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-20-2', "note": "", "level": 0},
            ],
        },
    },
    22: {
        "symbol": "Ti",
        "config": "[Ar] 3d2 4s2",
        "multiplicity": 3,
        "n_electrons": 22,
        "n_alpha": 12,
        "n_beta": 10,
        "n_closed": 10,
        "n_open": 2,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 6.8281, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ti&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.98, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    23: {
        "symbol": "V",
        "config": "[Ar] 3d3 4s2",
        "multiplicity": 4,
        "n_electrons": 23,
        "n_alpha": 13,
        "n_beta": 10,
        "n_closed": 10,
        "n_open": 3,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
                "max_iterations": 300,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 6.7462, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=V&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.92, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 6.866, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-62-2', "note": "", "level": 0},
            ],
        },
    },
    24: {
        "symbol": "Cr",
        "config": "[Ar] 3d5 4s1",
        "multiplicity": 7,
        "n_electrons": 24,
        "n_alpha": 15,
        "n_beta": 9,
        "n_closed": 9,
        "n_open": 6,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 6.7665, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cr&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.79, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    25: {
        "symbol": "Mn",
        "config": "[Ar] 3d5 4s2",
        "multiplicity": 6,
        "n_electrons": 25,
        "n_alpha": 15,
        "n_beta": 10,
        "n_closed": 10,
        "n_open": 5,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.434, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mn&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.66, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    26: {
        "symbol": "Fe",
        "config": "[Ar] 3d6 4s2",
        "multiplicity": 5,
        "n_electrons": 26,
        "n_alpha": 15,
        "n_beta": 11,
        "n_closed": 11,
        "n_open": 4,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.9024, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Fe&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.65, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    27: {
        "symbol": "Co",
        "config": "[Ar] 3d7 4s2",
        "multiplicity": 4,
        "n_electrons": 27,
        "n_alpha": 15,
        "n_beta": 12,
        "n_closed": 12,
        "n_open": 3,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
                "max_iterations": 300,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
                "max_iterations": 300,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.881, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Co&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.5, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    28: {
        "symbol": "Ni",
        "config": "[Ar] 3d8 4s2",
        "multiplicity": 3,
        "n_electrons": 28,
        "n_alpha": 15,
        "n_beta": 13,
        "n_closed": 13,
        "n_open": 2,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.6398, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ni&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.42, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    29: {
        "symbol": "Cu",
        "config": "[Ar] 3d10 4s1",
        "multiplicity": 2,
        "n_electrons": 29,
        "n_alpha": 15,
        "n_beta": 14,
        "n_closed": 14,
        "n_open": 1,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.7264, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cu&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 6.49, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 9.918, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-50-8', "note": "", "level": 0},
            ],
        },
    },
    30: {
        "symbol": "Zn",
        "config": "[Ar] 3d10 4s2",
        "multiplicity": 1,
        "n_electrons": 30,
        "n_alpha": 15,
        "n_beta": 15,
        "n_closed": 15,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 9.3942, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Zn&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 8.61, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [],
        },
    },
    31: {
        "symbol": "Ga",
        "config": "[Ar] 3d10 4s2 4p1",
        "multiplicity": 2,
        "n_electrons": 31,
        "n_alpha": 16,
        "n_beta": 15,
        "n_closed": 15,
        "n_open": 1,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 5.9993, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ga&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 5.59, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 5.731, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-55-3', "note": "", "level": 0},
            ],
        },
    },
    32: {
        "symbol": "Ge",
        "config": "[Ar] 3d10 4s2 4p2",
        "multiplicity": 3,
        "n_electrons": 32,
        "n_alpha": 17,
        "n_beta": 15,
        "n_closed": 15,
        "n_open": 2,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 7.8994, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ge&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 7.76, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 7.642, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-56-4', "note": "", "level": 0},
            ],
        },
    },
    33: {
        "symbol": "As",
        "config": "[Ar] 3d10 4s2 4p3",
        "multiplicity": 4,
        "n_electrons": 33,
        "n_alpha": 18,
        "n_beta": 15,
        "n_closed": 15,
        "n_open": 3,
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 9.7886, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=As&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 9.95, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 9.6, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-38-2', "note": "", "level": 0},
            ],
        },
    },
    34: {
        "symbol": "Se",
        "config": "[Ar] 3d10 4s2 4p4",
        "multiplicity": 3,
        "n_electrons": 34,
        "n_alpha": 18,
        "n_beta": 16,
        "n_closed": 16,
        "n_open": 2,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 9.7524, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Se&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 8.99, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 9.469, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7782-49-2', "note": "", "level": 0},
            ],
        },
    },
    35: {
        "symbol": "Br",
        "config": "[Ar] 3d10 4s2 4p5",
        "multiplicity": 2,
        "n_electrons": 35,
        "n_alpha": 18,
        "n_beta": 17,
        "n_closed": 17,
        "n_open": 1,
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 11.8138, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Br&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 11.23, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 11.547, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=10097-32-2', "note": "", "level": 0},
            ],
        },
    },
    36: {
        "symbol": "Kr",
        "config": "[Ar] 3d10 4s2 4p6",
        "multiplicity": 1,
        "n_electrons": 36,
        "n_alpha": 18,
        "n_beta": 18,
        "n_closed": 18,
        "n_open": 0,
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "SVWN"],
            }
        },
        "energies": {
            "experiment": [
                {"value": 13.9996, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Kr&units=1', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 14.26, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 13.752, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7439-90-9', "note": "", "level": 0},
            ],
        },
    },
    # ---- Period 5 --------------------------------------------------------
    # Not in scope of 6-31G** (covers H–Kr only).
    # 37: {
    #     "symbol": "Rb",
    #     "config": "[Kr] 5s1",
    #     "multiplicity": 2,
    #     "n_electrons": 37,
    #     "n_alpha": 19,
    #     "n_beta": 18,
    #     "n_closed": 18,
    #     "n_open": 1,
    #     "proposed_approach": {
    #         "ROHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         },
    #         "ROKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 4.1771, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Rb&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 3.98, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 38: {
    #     "symbol": "Sr",
    #     "config": "[Kr] 5s2",
    #     "multiplicity": 1,
    #     "n_electrons": 38,
    #     "n_alpha": 19,
    #     "n_beta": 19,
    #     "n_closed": 19,
    #     "n_open": 0,
    #     "proposed_approach": {
    #         "RHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "RKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 5.6949, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sr&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 5.03, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # Y not converging for ROHF.
    # 39: {
    #    "symbol": "Y", "config": "[Kr] 4d1 5s2", "multiplicity": 2,
    #    "n_electrons": 39, "n_alpha": 20, "n_beta": 19,
    #    "n_closed": 19, "n_open": 1,
    #    "proposed_approach": {
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 6.2173, "ionization_energies_hf_eV": 5.39,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Y&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # ZR problem in both: ROHF and UHF.
    # 40: {
    #    "symbol": "Zr", "config": "[Kr] 4d2 5s2", "multiplicity": 3,
    #    "n_electrons": 40, "n_alpha": 21, "n_beta": 19,
    #    "n_closed": 19, "n_open": 2,
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 6.6339, "ionization_energies_hf_eV": 5.70,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Zr&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # Not in scope of 6-31G** (covers H–Kr only).
    # 41: {
    #     "symbol": "Nb",
    #     "config": "[Kr] 4d4 5s1",
    #     "multiplicity": 6,
    #     "n_electrons": 41,
    #     "n_alpha": 23,
    #     "n_beta": 18,
    #     "n_closed": 18,
    #     "n_open": 5,
    #     "proposed_approach": {
    #         "ROHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         },
    #         "ROKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 6.7589, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Nb&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 5.82, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 42: {
    #     "symbol": "Mo",
    #     "config": "[Kr] 4d5 5s1",
    #     "multiplicity": 7,
    #     "n_electrons": 42,
    #     "n_alpha": 24,
    #     "n_beta": 18,
    #     "n_closed": 18,
    #     "n_open": 6,
    #     "proposed_approach": {
    #         "ROHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         },
    #         "ROKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 7.0924, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Mo&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 6.02, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 43: {
    #     "symbol": "Tc",
    #     "config": "[Kr] 4d5 5s2",
    #     "multiplicity": 6,
    #     "n_electrons": 43,
    #     "n_alpha": 24,
    #     "n_beta": 19,
    #     "n_closed": 19,
    #     "n_open": 5,
    #     "proposed_approach": {
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 7.119, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Tc&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 6.37, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 44: {
    #     "symbol": "Ru",
    #     "config": "[Kr] 4d7 5s1",
    #     "multiplicity": 5,
    #     "n_electrons": 44,
    #     "n_alpha": 24,
    #     "n_beta": 20,
    #     "n_closed": 20,
    #     "n_open": 4,
    #     "proposed_approach": {
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 7.3605, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ru&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 6.33, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 45: {
    #     "symbol": "Rh",
    #     "config": "[Kr] 4d8 5s1",
    #     "multiplicity": 4,
    #     "n_electrons": 45,
    #     "n_alpha": 24,
    #     "n_beta": 21,
    #     "n_closed": 21,
    #     "n_open": 3,
    #     "proposed_approach": {
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 7.4589, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Rh&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 6.3, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 46: {
    #     "symbol": "Pd",
    #     "config": "[Kr] 4d10",
    #     "multiplicity": 1,
    #     "n_electrons": 46,
    #     "n_alpha": 23,
    #     "n_beta": 23,
    #     "n_closed": 23,
    #     "n_open": 0,
    #     "proposed_approach": {
    #         "RHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/STO-3G.gbs",
    #             "max_iterations": 200,
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 8.3369, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Pd&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 7.4, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # Ag works for ROHF in 200 SCF iterations.
    # But it is not elegant solution.
    # 47: {
    #    "symbol": "Ag", "config": "[Kr] 4d10 5s1", "multiplicity": 2,
    #    "n_electrons": 47, "n_alpha": 24, "n_beta": 23,
    #    "n_closed": 23, "n_open": 1,
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 7.5762, "ionization_energies_hf_eV": 6.25,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Ag&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # Not in scope of 6-31G** (covers H–Kr only).
    # 48: {
    #     "symbol": "Cd",
    #     "config": "[Kr] 4d10 5s2",
    #     "multiplicity": 1,
    #     "n_electrons": 48,
    #     "n_alpha": 24,
    #     "n_beta": 24,
    #     "n_closed": 24,
    #     "n_open": 0,
    #     "proposed_approach": {
    #         "RHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "RKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 8.9938, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Cd&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 8.15, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # 49: {
    #     "symbol": "In",
    #     "config": "[Kr] 4d10 5s2 5p1",
    #     "multiplicity": 2,
    #     "n_electrons": 49,
    #     "n_alpha": 25,
    #     "n_beta": 24,
    #     "n_closed": 24,
    #     "n_open": 1,
    #     "proposed_approach": {
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 5.7864, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=In&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 5.27, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [],
    #     },
    # },
    # Not in scope of 6-31G** (covers H–Kr only).
    # 50: {
    #     "symbol": "Sn",
    #     "config": "[Kr] 4d10 5s2 5p2",
    #     "multiplicity": 3,
    #     "n_electrons": 50,
    #     "n_alpha": 26,
    #     "n_beta": 24,
    #     "n_closed": 24,
    #     "n_open": 2,
    #     "proposed_approach": {
    #         "UHF": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
    #             "max_iterations": 200,
    #         },
    #         "UKS": {
    #             "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31Gss.gbs",
    #             "max_iterations": 200,
    #             "functional": ["B3LYP", "PBE", "SVWN"],
    #         }
    #     },
    #     "energies": {
    #         "experiment": [
    #             {"value": 7.3439, "reference": 'https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sn&units=1', "note": "", "level": 0},
    #         ],
    #         "hf": [
    #             {"value": 7.16, "reference": 'https://doi.org/10.12691/wjce-5-3-6', "note": "", "level": 0},
    #         ],
    #         "dft": [
    #             {"value": 7.122, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7440-31-5', "note": "", "level": 0},
    #         ],
    #     },
    # },
    # Sb works for ROHF in 200 SCF iterations.
    # But it is not elegant solution.
    # 51: {
    #    "symbol": "Sb", "config": "[Kr] 4d10 5s2 5p3", "multiplicity": 4,
    #    "n_electrons": 51, "n_alpha": 27, "n_beta": 24,
    #    "n_closed": 24, "n_open": 3,
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 8.6084, "ionization_energies_hf_eV": 8.76,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sb&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # Te problem in both: ROHF and UHF.
    # 52: {
    #    "symbol": "Te", "config": "[Kr] 4d10 5s2 5p4", "multiplicity": 3,
    #    "n_electrons": 52, "n_alpha": 27, "n_beta": 25,
    #    "n_closed": 25, "n_open": 2,
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 9.0096, "ionization_energies_hf_eV": 8.24,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Te&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # I works for ROHF in 200 SCF iterations.
    # But it is not elegant solution.
    # 53: {
    #    "symbol": "I", "config": "[Kr] 4d10 5s2 5p5", "multiplicity": 2,
    #    "n_electrons": 53, "n_alpha": 27, "n_beta": 26,
    #    "n_closed": 26, "n_open": 1,
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 10.4513, "ionization_energies_hf_eV": 10.07,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=I&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
    # TO DO: 3-21G does not work for Xe.
    # Other Gaussian basis sets, does not contain Xe data.
    # Need to find proper basis set.
    # Check STO-6G as well, but Probably some relativistic effects may play a role here
    # 54: {
    #    "symbol": "Xe", "config": "[Kr] 4d10 5s2 5p6", "multiplicity": 1,
    #    "n_electrons": 54, "n_alpha": 27, "n_beta": 27,
    #    "n_closed": 27, "n_open": 0,
    #    "proposed_approach": {
    #        "RHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
    #            "max_iterations": 200,
    #        }
    #    },
    #    "ionization_energies_experiment_eV": 12.1298, "ionization_energies_hf_eV": 12.44,
    #    "experimental_reference_url": "https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Xe&units=1",
    #    "hf_reference_url": "https://doi.org/10.12691/wjce-5-3-6",
    # },
}

# MOLECULES_HOMO_ENERGIES  –  keyed by molecule identifier (str, e.g. "H2O").
# Entries are grouped by primary method: RHF (singlets), ROHF (open-shell
# where it converges), UHF (all open-shell systems).
# Molecule-specific fields:
#   "formula"        – molecular formula with Unicode superscripts
#   "name"           – common name ("" if none)
#   "technical_name" – SMILES string
#   "geometry"       – List[{"symbol": str, "x": float, "y": float, "z": float}]
#                      Cartesian coordinates in Å (experimental Re, z-axis
#                      along bond/principal axis; source: NIST CCCBDB)
# proposed_basis_set uses the largest Pople basis covering all constituent atoms.
MOLECULES: dict = {
    # ---- RHF molecules (closed-shell, multiplicity == 1) -----------------
    # Hydrogen molecule: simplest two-electron system; HOMO = 1σg
    "H2": {
        "formula": "H₂",
        "name": "hydrogen",
        "technical_name": "[HH]",
        "geometry": [
            {"symbol": "H", "x": 0.0000, "y": 0.0000, "z": 0.3707},
            {"symbol": "H", "x": 0.0000, "y": 0.0000, "z": -0.3707},
        ],
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 1,
        "n_electrons": 2,
        "n_alpha": 1,
        "n_beta": 1,
        "n_closed": 1,
        "n_open": 0,
        "energies": {
            "experiment": [
                {"value": 15.4259, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 16.17, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 15.452, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=1333-74-0', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Water: HOMO = 1b₁ (lone pair on O perpendicular to molecular plane)
    "H2O": {
        "formula": "H₂O",
        "name": "water",
        "technical_name": "O",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.0000},
            {"symbol": "H", "x": 0.7572, "y": 0.0000, "z": 0.5860},
            {"symbol": "H", "x": -0.7572, "y": 0.0000, "z": 0.5860},
        ],
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 1,
        "n_electrons": 10,
        "n_alpha": 5,
        "n_beta": 5,
        "n_closed": 5,
        "n_open": 0,
        "energies": {
            "experiment": [
                {"value": 12.621, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 13.79, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 11.524, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7732-18-5', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Dinitrogen: HOMO = 3σg (note: 3σg lies above 1πu in HF ordering)
    "N2": {
        "formula": "N₂",
        "name": "nitrogen",
        "technical_name": "N#N",
        "geometry": [
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": 0.5489},
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5489},
        ],
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 1,
        "n_electrons": 14,
        "n_alpha": 7,
        "n_beta": 7,
        "n_closed": 7,
        "n_open": 0,
        "energies": {
            "experiment": [
                {"value": 15.5808, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 16.71, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 15.428, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7727-37-9', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Carbon monoxide: HOMO = 5σ (weakly bonding, primarily C lone pair)
    "CO": {
        "formula": "CO",
        "name": "carbon monoxide",
        "technical_name": "[C-]#[O+]",
        "geometry": [
            {"symbol": "C", "x": 0.0000, "y": 0.0000, "z": -0.5641},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.5641},
        ],
        "proposed_approach": {
            "RHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "RKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 1,
        "n_electrons": 14,
        "n_alpha": 7,
        "n_beta": 7,
        "n_closed": 7,
        "n_open": 0,
        "energies": {
            "experiment": [
                {"value": 14.014, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C630080', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 15.05, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 14.029, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=630-08-0', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Dioxygen: triplet ground state; HOMO = 1πg (degenerate, singly occupied)
    # ionization_energies_hf_eV = −ε_α_HOMO (alpha-spin HOMO energy).  UHF introduces spin
    # contamination; <S²> may deviate from the exact S(S+1) value.
    # Dioxygen (UHF): same geometry as ROHF reference above
    "O2": {
        "formula": "O₂",
        "name": "oxygen",
        "technical_name": "[O][O]",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.6038},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": -0.6038},
        ],
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 3,
        "n_electrons": 16,
        "n_alpha": 9,
        "n_beta": 7,
        "n_closed": 7,
        "n_open": 2,
        "energies": {
            "experiment": [
                {"value": 12.0697, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 15.84, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 13.266, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=7782-44-7', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Nitric oxide: doublet; HOMO = 2π (one electron in antibonding π*)
    "NO": {
        "formula": "NO",
        "name": "nitric oxide",
        "technical_name": "[N]=O",
        "geometry": [
            {"symbol": "N", "x": 0.0000, "y": 0.0000, "z": -0.5754},
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": 0.5754},
        ],
        "proposed_approach": {
            "ROHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31++Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            },
            "ROKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "SVWN"],
            }
        },
        "multiplicity": 2,
        "n_electrons": 15,
        "n_alpha": 8,
        "n_beta": 7,
        "n_closed": 7,
        "n_open": 1,
        "energies": {
            "experiment": [
                {"value": 9.2643, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C10102439', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 11.51, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 10.121, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=10102-43-9', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Hydroxyl radical: doublet; HOMO = 1π (degenerate, one singly occupied)
    "OH": {
        "formula": "OH",
        "name": "hydroxyl",
        "technical_name": "[OH]",
        "geometry": [
            {"symbol": "O", "x": 0.0000, "y": 0.0000, "z": -0.4849},
            {"symbol": "H", "x": 0.0000, "y": 0.0000, "z": 0.4849},
        ],
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP"],
            }
        },
        "multiplicity": 2,
        "n_electrons": 9,
        "n_alpha": 5,
        "n_beta": 4,
        "n_closed": 4,
        "n_open": 1,
        "energies": {
            "experiment": [
                {"value": 13.017, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C3352576', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 13.02, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 12.354, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=3352-57-6', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
    },
    # Methyl radical: doublet D₃h; HOMO = 2a₂'' (singly occupied, p_z on C)
    "CH3": {
        "formula": "CH₃",
        "name": "methyl radical",
        "technical_name": "[CH3]",
        "geometry": [
            {"symbol": "C", "x": 0.0000, "y": 0.0000, "z": 0.0000},
            {"symbol": "H", "x": 1.0790, "y": 0.0000, "z": 0.0000},
            {"symbol": "H", "x": -0.5395, "y": 0.9344, "z": 0.0000},
            {"symbol": "H", "x": -0.5395, "y": -0.9344, "z": 0.0000},
        ],
        "proposed_approach": {
            "UHF": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/3-21G.gbs",
                "max_iterations": 200,
            },
            "UKS": {
                "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-311++Gss.gbs",
                "max_iterations": 200,
                "functional": ["B3LYP", "PBE", "SVWN"],
            }
        },
        "multiplicity": 2,
        "n_electrons": 9,
        "n_alpha": 5,
        "n_beta": 4,
        "n_closed": 4,
        "n_open": 1,
        "energies": {
            "experiment": [
                {"value": 9.84, "reference": 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C2229078', "note": "", "level": 0},
            ],
            "hf": [
                {"value": 11.4, "reference": 'https://cccbdb.nist.gov/', "note": "", "level": 0},
            ],
            "dft": [
                {"value": 9.834, "reference": 'https://cccbdb.nist.gov/ie2x.asp?casno=2229-07-4', "note": "B3LYP/3-21G adiabatic IE (delta-SCF)", "level": 0},
            ],
        },
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
    # CrO works for ROHF with 200 SCF iteration loop
    #"CrO": {
    #    "formula": "CrO",
    #    "name": "chromium monoxide",
    #    "technical_name": "[Cr]=O",
    #    "geometry": [
    #        {"symbol": "Cr", "x": 0.0000, "y": 0.0000, "z": -0.8075},
    #        {"symbol": "O",  "x": 0.0000, "y": 0.0000, "z":  0.8075},
    #    ],
    #    "proposed_approach": {
    #        "ROHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
    #            "max_iterations": 200,
    #         },
    #        "UHF": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
    #            "max_iterations": 200,
    #        },
    #        "UKS": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
    #            "max_iterations": 200,
    #            "functional": ["B3LYP", "PBE", "SVWN"],
    #        },
    #        "ROKS": {
    #            "proposed_basis_set": "q_block/compute/environment/constants/numerical/basis_set/pople/6-31G.gbs",
    #            "max_iterations": 200,
    #            "functional": ["B3LYP", "PBE", "SVWN"],
    #        },
    #    },
    #    "multiplicity": 5,
    #    "n_electrons": 32,
    #    "n_alpha": 18,
    #    "n_beta": 14,
    #    "n_closed": 14,
    #    "n_open": 4,
    #    "energies": {
    #        "experiment": [
    #            {"value": 8.16, "reference": "https://doi.org/10.1039/F29837902083", "note": "vertical IE, PE spectroscopy; adiabatic IE = 7.85 ± 0.02 eV", "level": 0},
    #        ],
    #        "hf": [
    #            {"value": 10.19, "reference": "https://cccbdb.nist.gov/", "note": "approximate near-basis-set-limit ROHF/UHF Koopmans; ±0.5 eV", "level": 0},
    #        ],
    #        "dft": [],
    #    },
    #},
}
