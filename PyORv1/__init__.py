"""
PyOR

Available modules:
- QuantumSystem
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
    "QuantumSystem",
    "Hamiltonian",
    "DensityMatrix",
    "QunObj",
    "HardPulse",
    "Basis",
    "Evolutions",
    "Plotting",
    "Spro",
]

# Version
__version__ = "1.0.0"

def info():
    """Print available PyOR modules."""
    print("PyOR loaded modules:")
    for name in __all__:
        print(f" - {name}")