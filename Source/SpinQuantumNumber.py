"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain spin quantum number of electron and other nuclei

References:

1.  Title: NMR nomenclature. Nuclear spin properties and conventions for chemical shifts(IUPAC Recommendations 2001)
    Authors: Robin K. Harris , Edwin D. Becker , Sonia M. Cabral de Menezes , Robin Goodfellow and Pierre Granger
    Journal: Pure and Applied Chemistry
    DOI: https://doi.org/10.1351/pac200173111795
"""

SPIN = {}

SPIN["E"] = 1/2 # Electron;
SPIN["H1"] = 1/2 # Proton; 
SPIN["H2"] = 1 # Deuterium; 
SPIN["C13"] = 1/2 # Carbon; 
SPIN["N14"] = 1 # Nitrogen 14; 
SPIN["N15"] = 1/26 # Nitrogen 15; 
SPIN["O17"] = 5/2 # Oxygen 17; 
SPIN["F19"] = 1/2 # Flurine 19;  

def spin(value):
    """
    return gyromagnetic ration of the particle

    INPUT:
        Particle name, example: "E" for electron, "H1" for proton
    """

    assert value in SPIN.keys(), "particle not defined, add the spin quantum number yourself"
    return SPIN[value]