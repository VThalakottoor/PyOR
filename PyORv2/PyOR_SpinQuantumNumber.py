"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides spin quantum numbers for the electron and a wide range 
    of nuclei used in magnetic resonance simulations.

    Units:
        Dimensionless (spin quantum number I)

    Convention:
        Element symbol first, then mass number (e.g., H1, C13, F19)

Reference:
    R. K. Harris et al., Pure Appl. Chem. 73 (2001) 1795–1818
    https://github.com/bennomeier/spindata/blob/master/spindata/spin.py
"""

SPIN = {}

# Electron
SPIN["E"] = 1/2  # Electron

# NV Center
SPIN["NV1"] = 1 # Spin 1

# --- Spin 1/2 Isotopes ---
SPIN["H1"] = 1/2   # Hydrogen-1 / Proton
SPIN["He3"] = 1/2  # Helium-3
SPIN["C13"] = 1/2  # Carbon-13
SPIN["N15"] = 1/2  # Nitrogen-15
SPIN["F19"] = 1/2  # Fluorine-19
SPIN["Si29"] = 1/2 # Silicon-29
SPIN["P31"] = 1/2  # Phosphorus-31
SPIN["Fe57"] = 1/2 # Iron-57
SPIN["Se77"] = 1/2 # Selenium-77
SPIN["Y89"] = 1/2  # Yttrium-89
SPIN["Rh103"] = 1/2 # Rhodium-103
SPIN["Ag107"] = 1/2 # Silver-107
SPIN["Ag109"] = 1/2 # Silver-109
SPIN["Cd111"] = 1/2 # Cadmium-111
SPIN["Sn115"] = 1/2 # Tin-115
SPIN["Sn117"] = 1/2 # Tin-117
SPIN["Sn119"] = 1/2 # Tin-119
SPIN["Te123"] = 1/2 # Tellurium-123
SPIN["Te125"] = 1/2 # Tellurium-125
SPIN["Xe129"] = 1/2 # Xenon-129
SPIN["W183"] = 1/2  # Tungsten-183
SPIN["Os187"] = 1/2 # Osmium-187
SPIN["Pt195"] = 1/2 # Platinum-195
SPIN["Hg199"] = 1/2 # Mercury-199
SPIN["Tl203"] = 1/2 # Thallium-203
SPIN["Tl205"] = 1/2 # Thallium-205
SPIN["Pb207"] = 1/2 # Lead-207

# --- Quadrupolar Isotopes ---
SPIN["H2"] = 1      # Hydrogen-2 / Deuterium
SPIN["Li6"] = 1     # Lithium-6
SPIN["Li7"] = 3/2   # Lithium-7
SPIN["Be9"] = 3/2   # Beryllium-9
SPIN["B10"] = 3     # Boron-10
SPIN["B11"] = 3/2   # Boron-11
SPIN["N14"] = 1     # Nitrogen-14
SPIN["O17"] = 5/2   # Oxygen-17
SPIN["Ne21"] = 3/2  # Neon-21
SPIN["Na23"] = 3/2  # Sodium-23
SPIN["Mg25"] = 5/2  # Magnesium-25
SPIN["Al27"] = 5/2  # Aluminum-27
SPIN["S33"] = 3/2   # Sulfur-33
SPIN["Cl35"] = 3/2  # Chlorine-35
SPIN["Cl37"] = 3/2  # Chlorine-37
SPIN["K39"] = 3/2   # Potassium-39
SPIN["K40"] = 4     # Potassium-40
SPIN["K41"] = 3/2   # Potassium-41
SPIN["Ca43"] = 7/2  # Calcium-43
SPIN["Sc45"] = 7/2  # Scandium-45
SPIN["Ti47"] = 5/2  # Titanium-47
SPIN["Ti49"] = 7/2  # Titanium-49
SPIN["V50"] = 6     # Vanadium-50
SPIN["V51"] = 7/2   # Vanadium-51
SPIN["Cr53"] = 3/2  # Chromium-53
SPIN["Mn55"] = 5/2  # Manganese-55
SPIN["Co59"] = 7/2  # Cobalt-59
SPIN["Ni61"] = 3/2  # Nickel-61
SPIN["Cu63"] = 3/2  # Copper-63
SPIN["Cu65"] = 3/2  # Copper-65
SPIN["Zn67"] = 5/2  # Zinc-67
SPIN["Ga69"] = 3/2  # Gallium-69
SPIN["Ga71"] = 3/2  # Gallium-71
SPIN["Ge73"] = 9/2  # Germanium-73
SPIN["As75"] = 3/2  # Arsenic-75
SPIN["Br79"] = 3/2  # Bromine-79
SPIN["Br81"] = 3/2  # Bromine-81
SPIN["Kr83"] = 9/2  # Krypton-83
SPIN["Rb85"] = 5/2  # Rubidium-85
SPIN["Rb87"] = 3/2  # Rubidium-87
SPIN["Sr87"] = 9/2  # Strontium-87
SPIN["Zr91"] = 5/2  # Zirconium-91
SPIN["Nb93"] = 9/2  # Niobium-93
SPIN["Mo95"] = 5/2  # Molybdenum-95
SPIN["Mo97"] = 5/2  # Molybdenum-97
SPIN["Tc99"] = 9/2  # Technetium-99
SPIN["Ru99"] = 5/2  # Ruthenium-99
SPIN["Ru101"] = 5/2 # Ruthenium-101
SPIN["Pd105"] = 5/2 # Palladium-105
SPIN["In113"] = 9/2 # Indium-113
SPIN["In115"] = 9/2 # Indium-115
SPIN["Sb121"] = 5/2 # Antimony-121
SPIN["Sb123"] = 7/2 # Antimony-123
SPIN["I127"] = 5/2  # Iodine-127
SPIN["Xe131"] = 3/2 # Xenon-131
SPIN["Cs133"] = 7/2 # Caesium-133
SPIN["Ba135"] = 3/2 # Barium-135
SPIN["Ba137"] = 3/2 # Barium-137
SPIN["La138"] = 5   # Lanthanum-138
SPIN["La139"] = 7/2 # Lanthanum-139
SPIN["Hf177"] = 7/2 # Hafnium-177
SPIN["Hf179"] = 9   # Hafnium-179
SPIN["Ta181"] = 7/2 # Tantalum-181
SPIN["Re185"] = 5/2 # Rhenium-185
SPIN["Re187"] = 5/2 # Rhenium-187
SPIN["Os189"] = 3/2 # Osmium-189
SPIN["Ir191"] = 3/2 # Iridium-191
SPIN["Ir193"] = 3/2 # Iridium-193
SPIN["Au197"] = 3/2 # Gold-197
SPIN["Hg201"] = 3/2 # Mercury-201
SPIN["Bi209"] = 9/2 # Bismuth-209

# --- Lanthanoids ---
SPIN["Pr141"] = 5/2 # Praseodymium-141
SPIN["Nd143"] = 7/2 # Neodymium-143
SPIN["Nd145"] = 7/2 # Neodymium-145
SPIN["Sm147"] = 7/2 # Samarium-147
SPIN["Sm149"] = 7/2 # Samarium-149
SPIN["Eu151"] = 5/2 # Europium-151
SPIN["Eu153"] = 5/2 # Europium-153
SPIN["Gd155"] = 3/2 # Gadolinium-155
SPIN["Gd157"] = 3/2 # Gadolinium-157
SPIN["Tb159"] = 3/2 # Terbium-159
SPIN["Dy161"] = 5/2 # Dysprosium-161
SPIN["Dy163"] = 5/2 # Dysprosium-163
SPIN["Ho165"] = 7/2 # Holmium-165
SPIN["Er167"] = 7/2 # Erbium-167
SPIN["Yb173"] = 5/2 # Ytterbium-173
SPIN["Lu175"] = 7/2 # Lutetium-175
SPIN["Lu176"] = 7   # Lutetium-176
SPIN["U235"] = 7/2  # Uranium-235


def spin(value):
    """
    Returns the spin quantum number of a specified particle.

    Parameters
    ----------
    value : str
        Particle symbol (e.g., "H1", "C13", "F19", "E")

    Returns
    -------
    float
        Spin quantum number
    """
    assert value in SPIN, "particle not defined, add the spin quantum number yourself"
    return SPIN[value]
