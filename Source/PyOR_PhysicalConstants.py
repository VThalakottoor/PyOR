"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain physical constants
"""

import numpy as np

CONSTANTS = {}

CONSTANTS["pl"] = 6.626e-34 # Planck Constant; J s
CONSTANTS["hbar"] = 1.05457182e-34 # Planck Constant; J s rad^-1
CONSTANTS["ep0"] = 8.854e-12 # Permitivity of free space; F m^-1
CONSTANTS["mu0"] = 4 * np.pi * 1.0e-7 # Permeabiltiy of free space; N A^-2 or H m^-1
CONSTANTS["kb"] = 1.380649e-23 # Boltzmann Constant; J K^-1
CONSTANTS["bm"] = 9.2740100657e-24 # Bohr magneton; J T^-1
CONSTANTS["nm"] = 5.0507837393e-27 # Nuclear magneton; J T^-1

def constants(value):
    """
    Returns the value of a fundamental physical constant.

    This function provides access to a dictionary of commonly used physical constants 
    in quantum mechanics, electromagnetism, and thermodynamics. The values are given 
    in SI units unless otherwise specified.

    Parameters:
    -----------
    value : str
        The key representing the desired constant. Available keys include:
            - "pl"   : Planck constant (h), in joule·seconds (J·s)
            - "hbar" : Reduced Planck constant (ħ = h / 2π), in joule·seconds per radian (J·s·rad⁻¹)
            - "ep0"  : Vacuum permittivity (ε₀), in farads per meter (F·m⁻¹)
            - "mu0"  : Vacuum permeability (μ₀), in newtons per ampere squared (N·A⁻²) or henries per meter (H·m⁻¹)
            - "kb"   : Boltzmann constant (k_B), in joules per kelvin (J·K⁻¹)
            - "bm"   : Bohr magneton (μ_B), in joules per tesla (J·T⁻¹)
            - "nm"   : Nuclear magneton (μ_N), in joules per tesla (J·T⁻¹)

    Returns:
    --------
    float
        The numerical value of the requested physical constant in SI units.

    Raises:
    -------
    AssertionError
        If the specified key is not found in the `CONSTANTS` dictionary.
        Users may add new constants manually if needed.

    Example:
    --------
    constants("pl")
    6.626e-34

    constants("kb")
    1.380649e-23
    """

    assert value in CONSTANTS.keys(), "physical constant not defined, add the physical constant yourself"
    return CONSTANTS[value]