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
from .PyOR_QuantumSystem import QuantumSystem
from .PyOR_QuantumSystems import QuantumSystems
from .PyOR_Hamiltonian import Hamiltonian
from .PyOR_DensityMatrix import DensityMatrix
from .PyOR_QuantumObject import QunObj
from .PyOR_HardPulse import HardPulse
from .PyOR_Basis import Basis
from .PyOR_Evolution import Evolutions
from .PyOR_Plotting import Plotting
from .PyOR_Relaxation import RelaxationProcess
from . import PyOR_SignalProcessing as Spro
from .PyOR_PhysicalConstants import constants
from .PyOR_Gamma import gamma
from .PyOR_QuantumLibrary import QuantumLibrary

# Public API
__all__ = [
    "QuantumSystem",
    "QuantumSystems",
    "Hamiltonian",
    "DensityMatrix",
    "QunObj",
    "HardPulse",
    "Basis",
    "Evolutions",
    "Plotting",
    "Spro",
    "constants",
    "gamma",
    "RelaxationProcess",
    "QuantumLibrary",
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
    print("Email: vineeth.thalakottoor@cea.fr\n")
    print('"Everybody can simulate Magnetic Resonance"\n')
          
    print("Imported modules:\n")

    print("* QuantumSystem          (QuantumSystem from PyOR_QuantumSystem - for single quantum system)")
    print("* QuantumSystems         (QuantumSystems from PyOR_QuantumSystems - for multiple quantum systems)")
    print("** Hamiltonian           (from PyOR_Hamiltonian)")
    print("** DensityMatrix         (from PyOR_DensityMatrix)")
    print("** HardPulse             (from PyOR_HardPulse)")
    print("** Basis                 (from PyOR_Basis)")
    print("** RelaxationProcess     (from PyOR_Relaxation)")
    print("** Evolutions            (from PyOR_Evolution)")
    print("** Plotting              (from PyOR_Plotting)")
    print("** Spro                  (from PyOR_SignalProcessing)")
    print("** constants             (from PyOR_PhysicalConstants)")
    print("** gamma                 (from PyOR_Gamma)")
    print("* QunObj                 (from PyOR_QuantumObject)")
    print("* QuantumLibrary         (from PyOR_QuantumLibrary)")

    print('\nHow to start?')
    print('\nMake a spin list like, Spin_list = {"A" : "H1", "B" : "H1"}')
    print('\nThen create an object, QS = QunS(Spin_list)')
    print('\nCall modules: QS.Hamiltonian, QS.DensityMatrix, QS.HardPulse, QS.Basis, QS.RelaxationProcess, QS.Evolutions, QS.Plotting, QS.Spro, QS.constants, QS.gamma')


# Version
__version__ = "2.0.0"

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
