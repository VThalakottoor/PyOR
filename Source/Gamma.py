"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain gyromagnetic ratio of electron and other nuclei

References:

1.  Title: NMR nomenclature. Nuclear spin properties and conventions for chemical shifts(IUPAC Recommendations 2001)
    Authors: Robin K. Harris , Edwin D. Becker , Sonia M. Cabral de Menezes , Robin Goodfellow and Pierre Granger
    Journal: Pure and Applied Chemistry
    DOI: https://doi.org/10.1351/pac200173111795
"""

GAMMA = {}

GAMMA["E"] = -1.761e11 # Electron; rad s^-1 T^-1
GAMMA["H1"] = 267.522e6 # Proton; rad s^-1 T^-1
GAMMA["H2"] = 41.065e6 # Deuterium; rad s^-1 T^-1
GAMMA["C13"] = 67.2828e6 # Carbon; rad s^-1 T^-1
GAMMA["N14"] = 19.311e6 # Nitrogen 14; rad s^-1 T^-1
GAMMA["N15"] = -27.116e6 # Nitrogen 15; rad s^-1 T^-1
GAMMA["O17"] = -36.264e6 # Oxygen 17; rad s^-1 T^-1
GAMMA["F19"] = 251.815e6 # Flurine 19; rad s^-1 T^-1 

def gamma(value):
    """
    return gyromagnetic ration of the particle

    INPUT:
        Particle name, example: "E" for electron, "H1" for proton
    """

    assert value in GAMMA.keys(), "particle not defined, add the gyromagnetic ratio yourself"
    return GAMMA[value]