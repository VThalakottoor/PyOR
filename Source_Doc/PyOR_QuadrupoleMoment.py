"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain quadrapole moment (Q) values of electron and other nuclei
"""

QUADRUPOLE = {} # Unit: fm^2, values from ref 2

QUADRUPOLE["E"] = 0 # Electron; 
QUADRUPOLE["H1"] = 0 # Proton; 
QUADRUPOLE["H2"] = 0.285783 # Deuterium;
QUADRUPOLE["C13"] = 0 # Carbon; 
QUADRUPOLE["N14"] = 2.044 # Nitrogen 14; 
QUADRUPOLE["N15"] = 0 # Nitrogen 15; 
QUADRUPOLE["O17"] = -2.558 # Oxygen 17; 
QUADRUPOLE["F19"] = 0 # Flurine 19;  

def quadrupole(value):
    """
    Returns the nuclear electric quadrupole moment of a specified particle.

    The nuclear quadrupole moment is a measure of the non-spherical distribution 
    of electric charge within a nucleus. It plays a critical role in quadrupolar 
    interactions observed in NMR and other spectroscopic techniques, especially 
    for nuclei with spin quantum number I ≥ 1.

    Parameters:
    -----------
    value : str
        Symbol representing the particle. Common examples include:
            - "E"   : Electron
            - "H1"  : Proton (Hydrogen-1)
            - "H2"  : Deuterium (Hydrogen-2)
            - "C13" : Carbon-13
            - "N14" : Nitrogen-14
            - "N15" : Nitrogen-15
            - "O17" : Oxygen-17
            - "F19" : Fluorine-19

    Returns:
    --------
    float
        Nuclear electric quadrupole moment (fm²).

    Raises:
    -------
    AssertionError
        If the particle symbol is not found in the predefined `QUADRUPOLE` dictionary.
        Users can extend the dictionary by adding values manually as needed.

    Example:
    --------
    quadrupole("H2")
    0.285783

    quadrupole("N14")
    2.044

    Reference:
    ----------
    1. Harris, R. K., Becker, E. D., de Menezes, S. M. C., Goodfellow, R., & Granger, P. (2001).  
    NMR nomenclature. Nuclear spin properties and conventions for chemical shifts (IUPAC Recommendations 2001).  
    *Pure and Applied Chemistry*, 73(11), 1795–1818.  
    DOI: https://doi.org/10.1351/pac200173111795

    2. Solid State NMR, Principles, Methods, and Applications,  Klaus Müller and Marco Geppi
    """

    assert value in QUADRUPOLE.keys(), "particle not defined, add the quadrupole value yourself"
    return QUADRUPOLE[value]