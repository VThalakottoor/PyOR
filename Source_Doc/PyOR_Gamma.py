"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain gyromagnetic ratio of electron and other nuclei

Documentation is done.
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
    Returns the gyromagnetic ratio (γ) of a specified particle.

    The gyromagnetic ratio is a fundamental physical constant that relates the 
    magnetic moment of a particle to its angular momentum. It plays a key role 
    in nuclear magnetic resonance (NMR), electron paramagnetic resonance (EPR), 
    and magnetic field interactions.

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
        Gyromagnetic ratio of the given particle, in units of radians per second per tesla (rad·s⁻¹·T⁻¹).

    Raises:
    -------
    AssertionError
        If the particle symbol is not found in the predefined `GAMMA` dictionary.
        In such cases, users are expected to define and add the gyromagnetic ratio manually.

    Example:
    --------
    gamma("H1")
    267522000.0

    gamma("N15")
    -27116000.0

    Reference:
    ----------
    Harris, R. K., Becker, E. D., de Menezes, S. M. C., Goodfellow, R., & Granger, P. (2001).  
    NMR nomenclature. Nuclear spin properties and conventions for chemical shifts (IUPAC Recommendations 2001).  
    *Pure and Applied Chemistry*, 73(11), 1795–1818.  
    DOI: https://doi.org/10.1351/pac200173111795
    """

    assert value in GAMMA.keys(), "particle not defined, add the gyromagnetic ratio yourself"
    return GAMMA[value]