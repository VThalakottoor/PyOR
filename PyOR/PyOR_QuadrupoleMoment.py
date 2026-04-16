"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides electric quadrupole moment (Q) values for the electron
    and a wide range of nuclei relevant to magnetic resonance simulations.

    The quadrupole moment is important for modeling quadrupolar interactions
    in NMR and EPR experiments, especially for nuclei with spin > 1/2.

    Units:
        m^2

    Convention:
        Element symbol first, then mass number (e.g., H2, O17, La139)

Reference:
    1. R. K. Harris et al., NMR nomenclature. Nuclear spin properties and
       conventions for chemical shifts (IUPAC Recommendations 2001)
       Pure Appl. Chem., 73(11), 1795–1818, 2001.
       https://doi.org/10.1351/pac200173111795
       
    2. https://github.com/bennomeier/spindata/blob/master/spindata/quadrupolemoment.py
    
    2. Solid State NMR, Principles, Methods, and Applications,  Klaus Müller and Marco Geppi 
"""

QUADRUPOLE = {}

# Electron and common spin-1/2 nuclei
QUADRUPOLE["E"] = 0.0         # Electron
QUADRUPOLE["H1"] = 0.0        # Hydrogen-1 / Proton
QUADRUPOLE["C13"] = 0.0       # Carbon-13
QUADRUPOLE["N15"] = 0.0       # Nitrogen-15
QUADRUPOLE["F19"] = 0.0       # Fluorine-19

# --- Quadrupolar Isotopes ---
QUADRUPOLE["H2"] = 0.2860e-28     # Hydrogen-2 / Deuterium
QUADRUPOLE["Li6"] = -0.0808e-28   # Lithium-6
QUADRUPOLE["Li7"] = -4.01e-28     # Lithium-7
QUADRUPOLE["Be9"] = 5.288e-28     # Beryllium-9
QUADRUPOLE["B10"] = 8.459e-28     # Boron-10
QUADRUPOLE["B11"] = 4.059e-28     # Boron-11
QUADRUPOLE["N14"] = 2.044e-28     # Nitrogen-14
QUADRUPOLE["O17"] = -2.558e-28    # Oxygen-17
QUADRUPOLE["Ne21"] = 10.155e-28   # Neon-21
QUADRUPOLE["Na23"] = 10.4e-28     # Sodium-23
QUADRUPOLE["Mg25"] = 19.94e-28    # Magnesium-25
QUADRUPOLE["Al27"] = 14.66e-28    # Aluminum-27
QUADRUPOLE["S33"] = -6.78e-28     # Sulfur-33
QUADRUPOLE["Cl35"] = -8.165e-28   # Chlorine-35
QUADRUPOLE["Cl37"] = -6.435e-28   # Chlorine-37
QUADRUPOLE["K39"] = 5.85e-28      # Potassium-39
QUADRUPOLE["K40"] = -7.3e-28      # Potassium-40
QUADRUPOLE["K41"] = 7.11e-28      # Potassium-41
QUADRUPOLE["Ca43"] = -4.08e-28    # Calcium-43
QUADRUPOLE["Sc45"] = -22.0e-28    # Scandium-45
QUADRUPOLE["Ti47"] = 30.2e-28     # Titanium-47
QUADRUPOLE["Ti49"] = 24.7e-28     # Titanium-49
QUADRUPOLE["V50"] = 21.0e-28      # Vanadium-50
QUADRUPOLE["V51"] = -5.2e-28      # Vanadium-51
QUADRUPOLE["Cr53"] = -15.0e-28    # Chromium-53
QUADRUPOLE["Mn55"] = 33.0e-28     # Manganese-55
QUADRUPOLE["Co59"] = 42.0e-28     # Cobalt-59
QUADRUPOLE["Ni61"] = 16.2e-28     # Nickel-61
QUADRUPOLE["Cu63"] = -22.0e-28    # Copper-63
QUADRUPOLE["Cu65"] = -20.4e-28    # Copper-65
QUADRUPOLE["Zn67"] = 15.0e-28     # Zinc-67
QUADRUPOLE["Ga69"] = 17.1e-28     # Gallium-69
QUADRUPOLE["Ga71"] = 10.7e-28     # Gallium-71
QUADRUPOLE["Ge73"] = -19.6e-28    # Germanium-73
QUADRUPOLE["As75"] = 31.4e-28     # Arsenic-75
QUADRUPOLE["Br79"] = 31.3e-28     # Bromine-79
QUADRUPOLE["Br81"] = 26.2e-28     # Bromine-81
QUADRUPOLE["Kr83"] = 25.9e-28     # Krypton-83
QUADRUPOLE["Rb85"] = 27.6e-28     # Rubidium-85
QUADRUPOLE["Rb87"] = 13.35e-28    # Rubidium-87
QUADRUPOLE["Sr87"] = 33.5e-28     # Strontium-87
QUADRUPOLE["Zr91"] = -17.6e-28    # Zirconium-91
QUADRUPOLE["Nb93"] = -32.0e-28    # Niobium-93
QUADRUPOLE["Mo95"] = -2.2e-28     # Molybdenum-95
QUADRUPOLE["Mo97"] = 25.5e-28     # Molybdenum-97
QUADRUPOLE["Tc99"] = -12.9e-28    # Technetium-99
QUADRUPOLE["Ru99"] = 7.9e-28      # Ruthenium-99
QUADRUPOLE["Ru101"] = 45.7e-28    # Ruthenium-101
QUADRUPOLE["Pd105"] = 66.0e-28    # Palladium-105
QUADRUPOLE["In113"] = 79.9e-28    # Indium-113
QUADRUPOLE["In115"] = 81.0e-28    # Indium-115
QUADRUPOLE["Sb121"] = -36.0e-28   # Antimony-121
QUADRUPOLE["Sb123"] = -49.0e-28   # Antimony-123
QUADRUPOLE["I127"] = -71.0e-28    # Iodine-127
QUADRUPOLE["Xe131"] = -11.4e-28   # Xenon-131
QUADRUPOLE["Cs133"] = -0.343e-28  # Caesium-133
QUADRUPOLE["Ba135"] = 16.0e-28    # Barium-135
QUADRUPOLE["Ba137"] = 24.5e-28    # Barium-137
QUADRUPOLE["La138"] = 45.0e-28    # Lanthanum-138
QUADRUPOLE["La139"] = 20.0e-28    # Lanthanum-139
QUADRUPOLE["Hf177"] = 336.5e-28   # Hafnium-177
QUADRUPOLE["Hf179"] = 379.3e-28   # Hafnium-179
QUADRUPOLE["Ta181"] = 317.0e-28   # Tantalum-181
QUADRUPOLE["Re185"] = 218.0e-28   # Rhenium-185
QUADRUPOLE["Re187"] = 207.0e-28   # Rhenium-187
QUADRUPOLE["Os189"] = 85.6e-28    # Osmium-189
QUADRUPOLE["Ir191"] = 81.6e-28    # Iridium-191
QUADRUPOLE["Ir193"] = 75.1e-28    # Iridium-193
QUADRUPOLE["Au197"] = 54.7e-28    # Gold-197
QUADRUPOLE["Hg201"] = 38.6e-28    # Mercury-201
QUADRUPOLE["Bi209"] = -51.6e-28   # Bismuth-209

# --- Lanthanoids ---
QUADRUPOLE["Pr141"] = -5.89e-28   # Praseodymium-141
QUADRUPOLE["Nd143"] = -63.0e-28   # Neodymium-143
QUADRUPOLE["Nd145"] = -33.0e-28   # Neodymium-145
QUADRUPOLE["Sm147"] = -25.9e-28   # Samarium-147
QUADRUPOLE["Sm149"] = 7.4e-28     # Samarium-149
QUADRUPOLE["Eu151"] = 90.3e-28    # Europium-151
QUADRUPOLE["Eu153"] = 241.2e-28   # Europium-153
QUADRUPOLE["Gd155"] = 127.0e-28   # Gadolinium-155
QUADRUPOLE["Gd157"] = 135.0e-28   # Gadolinium-157
QUADRUPOLE["Tb159"] = 143.2e-28   # Terbium-159
QUADRUPOLE["Dy161"] = 250.7e-28   # Dysprosium-161
QUADRUPOLE["Dy163"] = 264.8e-28   # Dysprosium-163
QUADRUPOLE["Ho165"] = 358.0e-28   # Holmium-165
QUADRUPOLE["Er167"] = 356.5e-28   # Erbium-167
QUADRUPOLE["Yb173"] = 280.0e-28   # Ytterbium-173
QUADRUPOLE["Lu175"] = 349.0e-28   # Lutetium-175
QUADRUPOLE["Lu176"] = 497.0e-28   # Lutetium-176
QUADRUPOLE["U235"] = 493.6e-28    # Uranium-235


def quadrupole(value):
    """
    Returns the electric quadrupole moment of a specified particle or nucleus.

    The electric quadrupole moment is a measure of the non-spherical charge
    distribution within a nucleus. It is important for describing quadrupolar
    interactions in magnetic resonance, especially for nuclei with spin > 1/2.

    Parameters
    ----------
    value : str
        Particle symbol (e.g., "H2", "N14", "O17", "E")

    Returns
    -------
    float
        Electric quadrupole moment in m^2

    Raises
    ------
    AssertionError
        If the particle symbol is not found in the predefined `QUADRUPOLE`
        dictionary.
    """
    assert value in QUADRUPOLE, "particle not defined, add the quadrupole value yourself"
    return QUADRUPOLE[value]
    
    
    

