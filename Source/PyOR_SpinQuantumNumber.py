"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain spin quantum number of electron and other nuclei
"""

SPIN = {}

SPIN["E"] = 1/2 # Electron;
SPIN["H1"] = 1/2 # Proton; 
SPIN["H2"] = 1 # Deuterium; 
SPIN["C13"] = 1/2 # Carbon; 
SPIN["N14"] = 1 # Nitrogen 14; 
SPIN["N15"] = 1/2 # Nitrogen 15; 
SPIN["O17"] = 5/2 # Oxygen 17; 
SPIN["F19"] = 1/2 # Flurine 19;  

def spin(value):
    """
    Returns the spin quantum number of a specified particle.

    This function looks up the spin quantum number for common particles 
    (nuclei and electrons) based on predefined values. These spin quantum 
    numbers are commonly used in NMR, EPR, and quantum mechanical calculations.

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
        The spin quantum number of the given particle.

    Raises:
    -------
    AssertionError
        If the particle symbol is not found in the predefined `SPIN` dictionary.
        In that case, the user is expected to define and add the spin value manually.       

    Example:
    --------
    spin("H1")
    0.5

    spin("O17")
    2.5

    Notes:
    ------
    The `SPIN` dictionary can be extended to include other nuclei or particles as needed.

    References:
    ----------

    Title: NMR nomenclature. Nuclear spin properties and conventions for chemical shifts(IUPAC Recommendations 2001)
    Authors: Robin K. Harris , Edwin D. Becker , Sonia M. Cabral de Menezes , Robin Goodfellow and Pierre Granger
    Journal: Pure and Applied Chemistry
    DOI: https://doi.org/10.1351/pac200173111795     
    """

    assert value in SPIN.keys(), "particle not defined, add the spin quantum number yourself"
    return SPIN[value]
