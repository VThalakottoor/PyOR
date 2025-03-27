"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain physical constants
"""

import numpy as np

CONSTANTS = {}

CONSTANTS["pl"] = 6.626e-34 # Planck Constant; J s
CONSTANTS["hbar"] = 1.054e-34 # Planck Constant; J s rad^-1
CONSTANTS["ep0"] = 8.854e-12 # Permitivity of free space; F m^-1
CONSTANTS["mu0"] = 4 * np.pi * 1.0e-7 # Permeabiltiy of free space; N A^-2 or H m^-1
CONSTANTS["kb"] = 1.380e-23 # Boltzmann Constant; J K^-1

def constants(value):
    """
    return gyromagnetic ration of the particle

    INPUT:
        Particle name, example: "E" for electron, "H1" for proton
    """

    assert value in CONSTANTS.keys(), "physical constant not defined, add the physical constant yourself"
    return CONSTANTS[value]