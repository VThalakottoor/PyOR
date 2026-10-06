"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This file defines the `Particle` class, which represents a quantum particle 
    with properties relevant to magnetic resonance simulations.

Attributes:
    name (str): 
        The name of the particle (e.g., '1H', '13C', 'Electron').
    spin (float): 
        The spin quantum number of the particle.
    gamma (float): 
        The gyromagnetic ratio of the particle (in rad/s/T).
    quadrupole (float): 
        The quadrupole moment of the particle (if applicable, otherwise zero).
"""


try:
    from . import PyOR_SpinQuantumNumber
    from . import PyOR_Gamma
    from . import PyOR_QuadrupoleMoment
except ImportError:
    import PyOR_SpinQuantumNumber
    import PyOR_Gamma
    import PyOR_QuadrupoleMoment


class particle():
    def __init__(self, value):
        self.name = value
        self.spin = PyOR_SpinQuantumNumber.spin(value)
        self.gamma = PyOR_Gamma.gamma(value)
        self.quadrupole = PyOR_QuadrupoleMoment.quadrupole(value)
