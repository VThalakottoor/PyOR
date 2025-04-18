"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contains the NonLinear class, which provides utilities for incorporating 
non-linear effects such as radiation damping, dipolar shifts, and Gaussian noise 
into spin dynamics simulations in NMR.

Documentation is done.
"""

import numpy as np
from numpy import linalg as lina
import re
from IPython.display import display, Latex, Math
from sympy.physics.quantum.cg import CG

from PyOR_QuantumObject import QunObj

class NonLinear:
    def __init__(self, class_QS):
        """
        Initialize the NonLinear class with parameters from the quantum system.

        Parameters
        ----------
        class_QS : object
            An instance of the quantum system class which provides simulation parameters.
        """
        self.class_QS = class_QS
        self.NGaussian = self.class_QS.NGaussian
        self.Rdamping = self.class_QS.Rdamping
        self.Sx = self.class_QS.Sx_
        self.Sy = self.class_QS.Sy_
        self.Sz = self.class_QS.Sz_
        self.RDphase = self.class_QS.RDphase
        self.RDxi = self.class_QS.RDxi
        self.N_mean = self.class_QS.N_mean
        self.N_std = self.class_QS.N_std
        self.N_length = self.class_QS.N_length

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Radiation Damping and Dipolar Effects
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    def Noise_Gaussian(self, N_mean, N_std, N_length):
        """
        Generate Gaussian noise samples.

        Parameters
        ----------
        N_mean : float
            Mean of the Gaussian distribution.
        N_std : float
            Standard deviation of the Gaussian distribution.
        N_length : int
            Number of random samples to generate.

        Returns
        -------
        np.ndarray
            An array of random noise values or zeros depending on configuration.
        """
        if self.NGaussian:
            return np.random.normal(N_mean, N_std, N_length)
        else:
            return np.zeros((N_length))     

    def DipoleShift(self, rho):
        """
        Compute dipolar shift from the current density matrix.

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of the system.

        Returns
        -------
        float
            Average dipolar field shift along z-direction.
        """
        BavgD = 0.0
        if self.class_QS.Dipole_Shift:
            BavgD += self.class_QS.Shift_para * np.trace(np.matmul(np.sum(self.Sz, axis=0), rho))
        return BavgD    

    def Radiation_Damping(self, rho):
        """
        Compute the radiation damping field based on the system state.

        This function calculates the transverse magnetization contributions
        to radiation damping and optionally adds spin noise in two different
        ways depending on simulation settings.

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of the spin system.

        Returns
        -------
        complex
            The effective radiation damping field including optional noise.
            Real part corresponds to the x-component, imaginary part to y-component.
        """
        Sx = self.Sx
        Sy = self.Sy

        Brd = 0.0
        RDnoise_x = 0.0
        RDnoise_y = 0.0
        RDnoise_amp = 0.0
        RDnoise_ph = 0.0

        if self.NGaussian:
            # Option 1: Add noise separately in real and imaginary components
            RDnoise_x = self.Noise_Gaussian(self.N_mean, self.N_std, self.N_length)
            RDnoise_y = self.Noise_Gaussian(self.N_mean, self.N_std, self.N_length)

            # Option 2: Add noise using amplitude and phase representation
            RDnoise_amp = self.Noise_Gaussian(self.N_mean, self.N_std, self.N_length)
            RDnoise_ph = np.random.uniform(low=0.0, high=360.0, size=(self.N_length))

        if self.Rdamping:
            for i in range(self.class_QS.Nspins):
                Brd_1 = -1j * self.RDxi[i] * np.trace(np.matmul(Sx[i] + 1j * Sy[i], rho))
                Brd_1 *= np.exp(-1j * np.pi * self.RDphase[i] / 180.0)
                Brd += Brd_1

        # Add either complex component noise or amplitude-phase based noise
        if False:
            return Brd + RDnoise_x + 1j * RDnoise_y
        else:
            return Brd + RDnoise_amp * np.exp(1j * RDnoise_ph)
