"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain class Particle

Attribute:
    name
    spin
    gamma
    quadrupole

Documentation is done.
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
