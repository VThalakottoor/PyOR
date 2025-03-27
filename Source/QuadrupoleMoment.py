"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain quadrapole value of electron and other nuclei

References:

1.  Title: NMR nomenclature. Nuclear spin properties and conventions for chemical shifts(IUPAC Recommendations 2001)
    Authors: Robin K. Harris , Edwin D. Becker , Sonia M. Cabral de Menezes , Robin Goodfellow and Pierre Granger
    Journal: Pure and Applied Chemistry
    DOI: https://doi.org/10.1351/pac200173111795
"""

QUADRUPOLE = {}

QUADRUPOLE["E"] = 0 # Electron; 
QUADRUPOLE["H1"] = 0 # Proton; 
QUADRUPOLE["H2"] = 0.2860e-28 # Deuterium;
QUADRUPOLE["C13"] = 0 # Carbon; 
QUADRUPOLE["N14"] = 2.044e-28 # Nitrogen 14; 
QUADRUPOLE["N15"] = 0 # Nitrogen 15; 
QUADRUPOLE["O17"] = -2.558e-28 # Oxygen 17; 
QUADRUPOLE["F19"] = 0 # Flurine 19;  

def quadrupole(value):
    """
    return gyromagnetic ration of the particle

    INPUT:
        Particle name, example: "E" for electron, "H1" for proton
    """

    assert value in QUADRUPOLE.keys(), "particle not defined, add the quadrupole value yourself"
    return QUADRUPOLE[value]