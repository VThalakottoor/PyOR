"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    Gyromagnetic ratios (γ) for the electron and a wide range of nuclei
    used in magnetic resonance (NMR, EPR).

    Units:
        rad s^-1 T^-1

    Convention:
        Element symbol first, then mass number (e.g., H1, C13, F19)

Reference:
    R. K. Harris et al., Pure Appl. Chem. 73 (2001) 1795–1818
    https://github.com/bennomeier/spindata/blob/master/spindata/gamma.py
"""

GAMMA = {}

# Electron
GAMMA["E"] = 1.7608597e11  # Electron

# --- Spin 1/2 Isotopes ---
GAMMA["H1"] = 26.7522128e7   # Hydrogen-1 / Proton
GAMMA["He3"] = -20.3801587e7 # Helium-3
GAMMA["C13"] = 6.728284e7    # Carbon-13
GAMMA["N15"] = -2.71261804e7 # Nitrogen-15
GAMMA["F19"] = 25.18148e7    # Fluorine-19
GAMMA["Si29"] = -5.3190e7    # Silicon-29
GAMMA["P31"] = 10.8394e7     # Phosphorus-31
GAMMA["Fe57"] = 0.8680624e7  # Iron-57
GAMMA["Se77"] = 5.1253857e7  # Selenium-77
GAMMA["Y89"] = -1.3162791e7  # Yttrium-89
GAMMA["Rh103"] = -0.8468e7   # Rhodium-103
GAMMA["Ag107"] = -1.0889181e7 # Silver-107
GAMMA["Ag109"] = -1.2518634e7 # Silver-109
GAMMA["Cd111"] = -5.6983131e7 # Cadmium-111
GAMMA["Cd113"] = -5.9609155e7 # Cadmium-113
GAMMA["Sn115"] = -8.8013e7    # Tin-115
GAMMA["Sn117"] = -9.58879e7   # Tin-117
GAMMA["Sn119"] = -10.0317e7   # Tin-119
GAMMA["Te123"] = -7.059098e7  # Tellurium-123
GAMMA["Te125"] = -8.5108404e7 # Tellurium-125
GAMMA["Xe129"] = -7.452103e7  # Xenon-129
GAMMA["W183"] = 1.1282403e7   # Tungsten-183
GAMMA["Os187"] = 0.6192895e7  # Osmium-187
GAMMA["Pt195"] = 5.8385e7     # Platinum-195
GAMMA["Hg199"] = 4.8457913e7  # Mercury-199
GAMMA["Tl203"] = 15.5393338e7 # Thallium-203
GAMMA["Tl205"] = 15.6921808e7 # Thallium-205
GAMMA["Pb207"] = 5.58046e7    # Lead-207

# --- Quadrupolar Isotopes ---
GAMMA["H2"] = 4.10662791e7    # Hydrogen-2 / Deuterium
GAMMA["Li6"] = 3.9371709e7    # Lithium-6
GAMMA["Li7"] = 10.3977013e7   # Lithium-7
GAMMA["Be9"] = -3.759666e7    # Beryllium-9
GAMMA["B10"] = 2.8746786e7    # Boron-10
GAMMA["B11"] = 8.5847044e7    # Boron-11
GAMMA["N14"] = 1.9337792e7    # Nitrogen-14
GAMMA["O17"] = -3.62808e7     # Oxygen-17
GAMMA["Ne21"] = -2.11308e7    # Neon-21
GAMMA["Na23"] = 7.0808493e7   # Sodium-23
GAMMA["Mg25"] = -1.63887e7    # Magnesium-25
GAMMA["Al27"] = 6.9762715e7   # Aluminum-27
GAMMA["As33"] = 2.055685e7    # Arsenic-33
GAMMA["Cl35"] = 2.624198e7    # Chlorine-35
GAMMA["Cl37"] = 2.184368e7    # Chlorine-37
GAMMA["K39"] = 1.2500608e7    # Potassium-39
GAMMA["K40"] = -1.5542854e7   # Potassium-40
GAMMA["K41"] = 0.68606808e7   # Potassium-41
GAMMA["Ca43"] = -1.803069e7   # Calcium-43
GAMMA["Sc45"] = 6.5087973e7   # Scandium-45
GAMMA["Ti47"] = -1.5105e7     # Titanium-47
GAMMA["Ti49"] = -1.51095e7    # Titanium-49
GAMMA["V50"] = 2.6706490e7    # Vanadium-50
GAMMA["V51"] = 7.0455117e7    # Vanadium-51
GAMMA["Cr53"] = -1.5152e7     # Chromium-53
GAMMA["Mn55"] = 6.6452546e7   # Manganese-55
GAMMA["Co59"] = 6.332e7       # Cobalt-59
GAMMA["Ni61"] = -2.3948e7     # Nickel-61
GAMMA["Cu63"] = 7.1117890e7   # Copper-63
GAMMA["Cu65"] = 7.60435e7     # Copper-65
GAMMA["Zn67"] = 1.676688e7    # Zinc-67
GAMMA["Ga69"] = 6.438855e7    # Gallium-69
GAMMA["Ga71"] = 8.181171e7    # Gallium-71
GAMMA["Ge73"] = -0.9360303e7  # Germanium-73
GAMMA["As75"] = 4.596163e7    # Arsenic-75
GAMMA["Br79"] = 6.725616e7    # Bromine-79
GAMMA["Br81"] = 7.249776e7    # Bromine-81
GAMMA["Kr83"] = -1.03310e7    # Krypton-83
GAMMA["Rb85"] = 2.5927050e7   # Rubidium-85
GAMMA["Rb87"] = 8.786400e7    # Rubidium-87
GAMMA["Sr87"] = -1.1639376e7  # Strontium-87
GAMMA["Zr91"] = -2.49743e7    # Zirconium-91
GAMMA["Nb93"] = 6.5674e7      # Niobium-93
GAMMA["Mo95"] = -1.751e7      # Molybdenum-95
GAMMA["Mo97"] = -1.788e7      # Molybdenum-97
GAMMA["Tc99"] = 6.046e7       # Technetium-99
GAMMA["Ru99"] = -1.229e7      # Ruthenium-99
GAMMA["Ru101"] = -1.377e7     # Ruthenium-101
GAMMA["Pd105"] = -1.23e7      # Palladium-105
GAMMA["In113"] = 5.8845e7     # Indium-113
GAMMA["In115"] = 5.8972e7     # Indium-115
GAMMA["Sb121"] = 6.4435e7     # Antimony-121
GAMMA["Sb123"] = 3.4892e7     # Antimony-123
GAMMA["I127"] = 5.389573e7    # Iodine-127
GAMMA["Xe131"] = 2.209076e7   # Xenon-131
GAMMA["Cs133"] = 3.5332539e7  # Caesium-133
GAMMA["Ba135"] = 2.67550e7    # Barium-135
GAMMA["Ba137"] = 2.99295e7    # Barium-137
GAMMA["La138"] = 3.557239e7   # Lanthanum-138
GAMMA["La139"] = 3.8083318e7  # Lanthanum-139
GAMMA["Hf177"] = 1.086e7      # Hafnium-177
GAMMA["Hf179"] = -0.6821e7    # Hafnium-179
GAMMA["Ta181"] = 3.2438e7     # Tantalum-181
GAMMA["Re185"] = 6.1057e7     # Rhenium-185
GAMMA["Re187"] = 6.1682e7     # Rhenium-187
GAMMA["Os189"] = 2.10713e7    # Osmium-189
GAMMA["Ir191"] = 0.4812e7     # Iridium-191
GAMMA["Ir193"] = 0.5227e7     # Iridium-193
GAMMA["Au197"] = 0.473060e7   # Gold-197
GAMMA["Hg201"] = -1.788769e7  # Mercury-201
GAMMA["Bi209"] = 4.3750e7     # Bismuth-209

# --- Lanthanoids ---
GAMMA["Pr141"] = 8.1907e7     # Praseodymium-141
GAMMA["Nd143"] = -1.457e7     # Neodymium-143
GAMMA["Nd145"] = -0.898e7     # Neodymium-145
GAMMA["Sm147"] = -1.115e7     # Samarium-147
GAMMA["Sm149"] = -0.9192e7    # Samarium-149
GAMMA["Eu151"] = 6.6510e7     # Europium-151
GAMMA["Eu153"] = 2.9369e7     # Europium-153
GAMMA["Gd155"] = -0.82132e7   # Gadolinium-155
GAMMA["Gd157"] = -1.0769e7    # Gadolinium-157
GAMMA["Tb159"] = 6.431e7      # Terbium-159
GAMMA["Dy161"] = -0.9201e7    # Dysprosium-161
GAMMA["Dy163"] = 1.289e7      # Dysprosium-163
GAMMA["Ho165"] = 53.710e7     # Holmium-165
GAMMA["Er167"] = -0.77157e7   # Erbium-167
GAMMA["Tm169"] = -2.218e7     # Thulium-169
GAMMA["Yb171"] = 4.7288e7     # Ytterbium-171
GAMMA["Yb173"] = -1.3025e7    # Ytterbium-173
GAMMA["Lu175"] = 3.0552e7     # Lutetium-175
GAMMA["Lu176"] = 2.1684e7     # Lutetium-176
GAMMA["U235"] = -0.52e7       # Uranium-235


def gamma(value):
    """
    Returns the gyromagnetic ratio (γ) of a specified particle.

    Parameters
    ----------
    value : str
        Particle symbol (e.g., "H1", "C13", "F19", "E")

    Returns
    -------
    float
        Gyromagnetic ratio in rad s^-1 T^-1
    """
    assert value in GAMMA, "particle not defined, add the gyromagnetic ratio yourself"
    return GAMMA[value]
