# PyOR - Quantum Simulation Toolkit

"""
PyOR

Available modules:
- QunS (QuantumSystem)
- Hamiltonian
- DensityMatrix
- QunObj
- HardPulse
- Basis
- Evolutions
- Plotting
- Spro (Signal Processing)
"""

# Core imports
from .PyOR_QuantumSystem import QuantumSystem as QunS
from .PyOR_Hamiltonian import Hamiltonian
from .PyOR_DensityMatrix import DensityMatrix
from .PyOR_QuantumObject import QunObj
from .PyOR_HardPulse import HardPulse
from .PyOR_Basis import Basis
from .PyOR_Evolution import Evolutions
from .PyOR_Plotting import Plotting
from . import PyOR_SignalProcessing as Spro

# Public API
__all__ = [
    "QunS",
    "Hamiltonian",
    "DensityMatrix",
    "QunObj",
    "HardPulse",
    "Basis",
    "Evolutions",
    "Plotting",
    "Spro",
    "info"
]

def info_():
    # Logo
    P = [
    "*****",
    "*   *",
    "*****",
    "*    ",
    "*    ",
    "     ",
    "     "
    ]

    y = [
    "      ",
    "      ",
    " *   *",
    "  * * ",
    "   *  ",
    "  *   ",
    " *    "
    ]

    O = [
    "*****",
    "*   *",
    "*   *",
    "*   *",
    "*****",
    "     ",
    "     "
    ]

    R = [
    "*****",
    "*   *",
    "*****",
    "* *  ",
    "*  * ",
    "     ",
    "     "
    ]    
    for i in range(7):
        print(P[i], y[i], O[i], R[i])
    print("Welcome to Python On Resonance (PyOR)\n")
    print("Author: Vineeth Thalakottoor, IE CNRS, LSDRM, CEA, Paris-Saclay\n")
    print("Email: vineethfrancis.physics@gmail.com\n")
    print('"Everybody can simulate Magnetic Resonance"\n')
          
    print("Imported modules:\n")

    print("* QunS            (QuantumSystem from PyOR_QuantumSystem)")
    print("\t** Hamiltonian     (from PyOR_Hamiltonian)")
    print("\t** DensityMatrix   (from PyOR_DensityMatrix)")
    print("\t** HardPulse       (from PyOR_HardPulse)")
    print("\t** Basis           (from PyOR_Basis)")
    print("\t** Evolutions      (from PyOR_Evolution)")
    print("\t** Plotting        (from PyOR_Plotting)")
    print("\t** Spro            (from PyOR_SignalProcessing)")
    print("* QunObj          (from PyOR_QuantumObject)")

    print('\nHow to start? Make a spin list like, Spin_list = {"A" : "H1", "B" : "H1"}')
    print('\nThen create an object, QS = QunS(Spin_list)')
    print('\nCall modules: Hamiltonian, DensityMatrix, HardPulse, basis, Evoultions, Plotting and Spro as QS.Hamiltonian, QS.DensityMatrix, so on.')


# Version
__version__ = "1.0.0"

def info():
    print("Welcome to PyOR - Python On Resonance")
    print("PyOR loaded modules:")
    for name in __all__:
        print(f" - {name}")

def auto_info():
    info_()

# optional trigger
import os
os.environ["PYOR_AUTO_INFO"] = "1"
if os.environ.get("PYOR_AUTO_INFO") == "1":
    auto_info()        