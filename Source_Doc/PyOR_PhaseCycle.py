"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contains the PhaseCycle class, which handles phase cycling operations 
including pulse phasing and receiver phase adjustments in magnetic resonance simulations.

Documentation is done.
"""

import numpy as np
from scipy.linalg import expm

try:
    from .PyOR_Commutators import Commutators as COM
    from .PyOR_QuantumObject import QunObj
except ImportError:
    from PyOR_Commutators import Commutators as COM
    from PyOR_QuantumObject import QunObj


class PhaseCycle:
    def __init__(self, class_QS):
        """
        Initialize the PhaseCycle class.

        Parameters
        ----------
        class_QS : object
            Instance of the quantum system that provides operator definitions and simulation context.
        """
        self.class_QS = class_QS
        self.class_COM = COM()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Phase Cycling Operations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    def Pulse_Phase(self, SxQ, SyQ, phase):
        """
        Construct a pulse operator along a defined phase direction.

        This generates a combined spin operator that simulates a pulse 
        along the direction defined by `phase`:
        Pulse Operator = cos(phase) * Sx + sin(phase) * Sy

        Parameters
        ----------
        SxQ : QunObj
            Sx spin operator (should be a list of matrices or summed operator).
        SyQ : QunObj
            Sy spin operator (same dimensions as SxQ).
        phase : float
            Desired phase angle in degrees.

        Returns
        -------
        QunObj
            Spin operator for rotation about the defined phase axis.

        Raises
        ------
        TypeError
            If inputs are not instances of QunObj.
        """
        if not isinstance(SxQ, QunObj) or not isinstance(SyQ, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        Sx = SxQ.data
        Sy = SyQ.data

        phase_rad = np.pi * phase / 180.0
        return QunObj(np.cos(phase_rad) * np.sum(Sx, axis=0) +
                      np.sin(phase_rad) * np.sum(Sy, axis=0))
    
    def Receiver_Phase(self, SxQ, SyQ, phase): 
        """
        Construct a receiver (detection) operator with a defined phase.

        This rotates the detection operator (Sx + iSy) by the specified receiver phase:
        Detection Operator = (Sx + i*Sy) * exp(i * phase)

        Parameters
        ----------
        SxQ : QunObj
            Sx spin operator.
        SyQ : QunObj
            Sy spin operator.
        phase : float
            Receiver phase in degrees.

        Returns
        -------
        QunObj
            Complex detection operator after phase adjustment.

        Raises
        ------
        TypeError
            If inputs are not instances of QunObj.
        """
        if not isinstance(SxQ, QunObj) or not isinstance(SyQ, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        Sx = SxQ.data
        Sy = SyQ.data

        phase_rad = np.pi * phase / 180.0
        return QunObj((np.sum(Sx, axis=0) + 1j * np.sum(Sy, axis=0)) *
                      np.exp(1j * phase_rad))
