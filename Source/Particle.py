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
"""

import SpinQuantumNumber 
import Gamma
import QuadrupoleMoment 

class particle():
    def __init__(self, value):
        self.name = value
        self.spin = SpinQuantumNumber.spin(value)
        self.gamma = Gamma.gamma(value)
        self.quadrupole = QuadrupoleMoment.quadrupole(value)