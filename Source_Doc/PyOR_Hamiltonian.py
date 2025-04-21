""" 
PyOR - Python On Resonance 
Author: Vineeth Francis Thalakottoor Jose Chacko 
Email: vineethfrancis.physics@gmail.com 

This file contains the Hamiltonian class, which includes routines for building various 
spin interaction Hamiltonians used in magnetic resonance simulations.

Documentation is done.
"""

import numpy as np
from numpy import linalg as lina
from scipy.interpolate import interp1d
from scipy.linalg import expm
from io import StringIO
from joblib import Parallel, delayed

import PyOR_PhysicalConstants
import PyOR_Rotation
import PyOR_SphericalTensors as ST
from PyOR_QuantumObject import QunObj
import PyOR_SignalProcessing as Spro

class Hamiltonian:
    def __init__(self, class_QS):
        """
        Initializes the Hamiltonian class with quantum system parameters.

        Parameters:
        -----------
        class_QS : QuantumSystem
            Instance of the QuantumSystem class containing all system-level parameters
            like spin operators, gyromagnetic ratios, B0 field, offsets, etc.
        """
        self.class_QS = class_QS
        self.DTYPE_C = self.class_QS.DTYPE_C  # Data type for complex matrices
        self.hbar = PyOR_PhysicalConstants.constants("hbar")
        self.mu0 = PyOR_PhysicalConstants.constants("mu0")

        self.Gamma = self.class_QS.Gamma
        self.B0 = self.class_QS.B0
        self.Offset = self.class_QS.Offset         
        self.LarmorF = self.LarmorFrequency()
        self.LARMOR_F = self.class_QS.LARMOR_F
        for i, key in enumerate(self.class_QS.SpinDic):
            self.LARMOR_F[key] = self.LarmorF[i]

        # Inverse of 2 Pi – often used to convert between angular frequency and Hz
        self.Inverse2PI = 1.0 / (2.0 * np.pi)

        self.InteractioTensor_AngularFrequency = True

    def Update(self):
        """
        Updates internal parameters from the QuantumSystem instance.
        Useful when the quantum system is modified externally and 
        the Hamiltonian needs to be resynchronized.
        """
        self.class_QS.Update()
        self.Gamma = self.class_QS.Gamma
        self.B0 = self.class_QS.B0
        self.Offset = self.class_QS.Offset         
        self.LarmorF = self.LarmorFrequency()

    def LarmorFrequency(self):
        """
        Computes the Larmor frequency (Ω₀) for each spin in the lab frame.

        Larmor frequency is given by:
            Ω₀ = -γ * B₀ - 2π * δ
        where:
            γ = gyromagnetic ratio,
            B₀ = magnetic field strength in Tesla,
            δ = chemical shift offset (in Hz)

        Returns:
        --------
        numpy.ndarray
            Larmor frequencies (angular frequencies) for each spin in the system.
        """
        Gamma = self.Gamma
        B0 = self.B0
        Offset = self.Offset
        
        W0 = np.zeros((self.class_QS.Nspins))
        gamma = np.asarray(Gamma)
        offset = np.asarray(Offset)

        for i in range(self.class_QS.Nspins):
            W0[i] = -1 * gamma[i] * B0 - 2 * np.pi * offset[i]

        if self.class_QS.print_Larmor:
            print("Larmor Frequency in MHz: ", W0 / (2.0 * np.pi * 1.0e6))

        self.LarmorF = W0
        return W0

    def Zeeman(self):
        """
        Constructs the Zeeman Hamiltonian in the laboratory frame.

        H_Z = Σ_i ω₀ᵢ * S_zᵢ
        where ω₀ᵢ is the Larmor frequency of the i-th spin,
        and S_zᵢ is the z-component spin operator.

        Returns:
        --------
        QunObj
            The Zeeman Hamiltonian represented as a quantum object.
        """
        LarmorF = self.LarmorF
        Sz = self.class_QS.Sz_

        Hz = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)
        for i in range(self.class_QS.Nspins):
            Hz += LarmorF[i] * Sz[i]

        return QunObj(Hz)

    def Zeeman_RotFrame(self):
        """
        Constructs the Zeeman Hamiltonian in the rotating frame.

        H_Z_RF = Σ_i (ω₀ᵢ - ω_RFᵢ) * S_zᵢ
        where:
            - ω₀ᵢ is the Larmor frequency of the i-th spin,
            - ω_RFᵢ is the rotating frame reference frequency.

        Returns:
        --------
        QunObj
            The Zeeman Hamiltonian in the rotating frame.
        """
        LarmorF = self.LarmorF
        OmegaRF = self.class_QS.OmegaRF
        Sz = self.class_QS.Sz_

        omegaRF = np.asarray(OmegaRF)
        Hz = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)
        for i in range(self.class_QS.Nspins):
            Hz += (LarmorF[i] - omegaRF[i]) * Sz[i]
        
        return QunObj(Hz)

    def Zeeman_B1(self, Omega1, Omega1Phase):  
        """
        Constructs the B₁ Hamiltonian for RF excitation in the rotating frame.

        Assumes a continuous wave RF field:
            H_RF = ω₁ * [Sₓ cos(ϕ) + Sᵧ sin(ϕ)]

        Parameters:
        -----------
        Omega1 : float
            RF amplitude in Hz (nutation frequency).
        Omega1Phase : float
            RF phase in degrees.

        Returns:
        --------
        QunObj
            Time-independent B₁ Hamiltonian in rotating frame.
        """
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 

        HzB1 = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)
        omega1 = 2 * np.pi * Omega1
        Omega1Phase = np.pi * Omega1Phase / 180.0  # convert degrees to radians

        for i in range(self.class_QS.Nspins):
            HzB1 += omega1 * (Sx[i] * np.cos(Omega1Phase) + Sy[i] * np.sin(Omega1Phase))

        return QunObj(HzB1)

    def Zeeman_B1_Offresonance(self, t, Omega1, Omega1freq, Omega1Phase):  
        """
        Constructs a time-dependent Zeeman Hamiltonian for an off-resonant RF field.

        H(t) = ω₁ * [Sₓ cos(ω_RF * t + ϕ) + Sᵧ sin(ω_RF * t + ϕ)]

        Parameters:
        -----------
        t : float
            Time in seconds.
        Omega1 : float
            RF amplitude (nutation frequency in Hz).
        Omega1freq : float
            RF frequency in Hz (off-resonance frequency in rotating frame).
        Omega1Phase : float
            Initial phase in degrees.

        Returns:
        --------
        numpy.ndarray
            Time-dependent Hamiltonian matrix (not a QunObj).
        """
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        
        HzB1 = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)
        omega1 = 2 * np.pi * Omega1
        Omega1freq = 2 * np.pi * Omega1freq
        Omega1Phase = np.pi * Omega1Phase / 180.0  # convert to radians

        for i in range(self.class_QS.Nspins):
            HzB1 += omega1 * (Sx[i] * np.cos(Omega1freq * t + Omega1Phase) +
                              Sy[i] * np.sin(Omega1freq * t + Omega1Phase))

        return HzB1

    def Zeeman_B1_ShapedPulse(self, t, Omega1T, Omega1freq, Omega1PhaseT):  
        """
        Constructs time-dependent B₁ Hamiltonian with shaped amplitude and phase (e.g., Gaussian).

        Parameters:
        -----------
        t : float
            Time (in seconds).
        Omega1T : callable
            Function returning time-varying RF amplitude (Hz).
        Omega1freq : float
            RF carrier frequency (Hz).
        Omega1PhaseT : callable
            Function returning time-varying phase (radians).

        Returns:
        --------
        numpy.ndarray
            Time-dependent Hamiltonian matrix (not a QunObj).
        """
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        
        HzB1 = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)

        omega1 = 2 * np.pi * Omega1T(t)
        Omega1freq = 2 * np.pi * Omega1freq
        Omega1Phase = Omega1PhaseT(t)

        for i in range(self.class_QS.Nspins):
            HzB1 += omega1 * (Sx[i] * np.cos(Omega1freq * t + Omega1Phase) +
                              Sy[i] * np.sin(Omega1freq * t + Omega1Phase))

        return HzB1

    def Jcoupling(self):    
        """
        Construct full J-coupling Hamiltonian (isotropic scalar coupling).

        H_J = ∑₍i<j₎ J_ij * (Sxᵢ·Sxⱼ + Syᵢ·Syⱼ + Szᵢ·Szⱼ)

        Inputs:
        -------
        J : 2D array (Hz)
            Symmetric matrix of J-couplings between spins i and j.
        Sx, Sy, Sz : list of ndarray
            Cartesian spin operators for all spins.

        Output:
        -------
        QunObj
            Full J-coupling Hamiltonian (angular frequency units).
        """
        J = self.class_QS.Jlist
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        Sz = self.class_QS.Sz_

        J = np.triu(2 * np.pi * J)  # Convert to angular frequency and zero lower triangle
        Hj = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)

        for i in range(self.class_QS.Nspins):
            for j in range(self.class_QS.Nspins):
                Hj += J[i][j] * (np.matmul(Sx[i], Sx[j]) +
                                 np.matmul(Sy[i], Sy[j]) +
                                 np.matmul(Sz[i], Sz[j]))

        return QunObj(Hj)

    def Jcoupling_Weak(self):    
        """
        Construct simplified J-coupling Hamiltonian (weak coupling approximation).

        H_J ≈ ∑₍i<j₎ J_ij * Szᵢ·Szⱼ

        Inputs:
        -------
        J : 2D array (Hz)
            Symmetric matrix of J-couplings between spins i and j.
        Sz : list of ndarray
            Z-component spin operators.

        Output:
        -------
        QunObj
            Simplified J-coupling Hamiltonian (angular frequency units).
        """
        J = self.class_QS.Jlist
        Sz = self.class_QS.Sz_

        J = np.triu(2 * np.pi * J)
        Hj = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)

        for i in range(self.class_QS.Nspins):
            for j in range(self.class_QS.Nspins):
                Hj += J[i][j] * np.matmul(Sz[i], Sz[j])

        return QunObj(Hj)

    def Dipole_Coupling_Constant(self, Gamma1, Gamma2, distance):
        """
        Compute the dipolar coupling constant between two spins.

        Formula:
        b_IS = - (μ₀ / 4π) * (γ₁ * γ₂ * ℏ) / r³

        Inputs:
        -------
        Gamma1 : float
            Gyromagnetic ratio of spin 1 (rad/T·s)
        Gamma2 : float
            Gyromagnetic ratio of spin 2 (rad/T·s)
        distance : float
            Inter-spin distance in meters

        Output:
        -------
        float
            Dipolar coupling constant in Hz
        """
        print("dipolar coupling constant (in Hz)")
        return self.mu0 * Gamma1 * Gamma2 * self.hbar * (distance ** -3) / (4 * np.pi) / (2 * np.pi)

    def DDcoupling(self):
        """
        Construct the Dipole-Dipole (DD) interaction Hamiltonian for all spin pairs.

        Supports:
        - Secular approximation (Hetero & Homo)
        - Full interaction (All terms: secular, pseudo-secular, non-secular)
        - Individual components: A, B, C, D, E, F

        Inputs from class_QS:
        ----------------------
        - DipolePairs: list of (i, j) spin index pairs
        - DipoleAngle: list of (theta, phi) for each pair in degrees
        - DipolebIS: list of dipolar coupling constants (Hz)
        - Dipole_DipolarAlpabet: interaction mode (secular, All, A–F)

        Output:
        -------
        QunObj
            Complete DD coupling Hamiltonian in angular frequency units
        """
        Sx, Sy, Sz = self.class_QS.Sx_, self.class_QS.Sy_, self.class_QS.Sz_
        Sp, Sm = self.class_QS.Sp_, self.class_QS.Sm_

        thetaAll, phiAll = np.array(self.class_QS.DipoleAngle).T
        thetaAll = np.pi * thetaAll / 180.0
        phiAll = np.pi * phiAll / 180.0

        Spin1, Spin2 = np.array(self.class_QS.DipolePairs).T
        Hdd = np.zeros((self.class_QS.Vdim, self.class_QS.Vdim), dtype=self.DTYPE_C)

        for i, j, mode, theta, phi, bIS in zip(Spin1, Spin2, self.class_QS.Dipole_DipolarAlpabet, thetaAll, phiAll, self.class_QS.DipolebIS):

            if mode == "secular Hetronuclear":
                A = -1 * np.matmul(Sz[i], Sz[j]) * (3 * np.cos(theta) ** 2 - 1)
                Hdd += 2 * np.pi * bIS * A

            elif mode == "secular Homonuclear":
                A = np.matmul(Sz[i], Sz[j]) * (3 * np.cos(theta) ** 2 - 1)
                B = 0.25 * (np.matmul(Sp[i], Sm[j]) + np.matmul(Sm[i], Sp[j])) * (3 * np.cos(theta) ** 2 - 1)
                Hdd += 2 * np.pi * bIS * (A + B)

            elif mode == "All":
                A = -1 * np.matmul(Sz[i], Sz[j]) * (3 * np.cos(theta) ** 2 - 1)
                B = 0.25 * (np.matmul(Sp[i], Sm[j]) + np.matmul(Sm[i], Sp[j])) * (3 * np.cos(theta) ** 2 - 1)
                C = (-3/2) * (np.matmul(Sp[i], Sz[j]) + np.matmul(Sz[i], Sp[j])) * np.sin(theta) * np.cos(theta) * np.exp(-1j * phi)
                D = (-3/2) * (np.matmul(Sm[i], Sz[j]) + np.matmul(Sz[i], Sm[j])) * np.sin(theta) * np.cos(theta) * np.exp(1j * phi)
                E = (-3/4) * np.matmul(Sp[i], Sp[j]) * np.sin(theta) ** 2 * np.exp(-1j * 2 * phi)
                F = (-3/4) * np.matmul(Sm[i], Sm[j]) * np.sin(theta) ** 2 * np.exp(1j * 2 * phi)
                Hdd += 2 * np.pi * bIS * (A + B + C + D + E + F)

            elif mode in ["A", "B", "C", "D", "E", "F"]:
                if mode == "A":
                    Hdd += 2 * np.pi * bIS * (-1 * np.matmul(Sz[i], Sz[j]) * (3 * np.cos(theta) ** 2 - 1))
                elif mode == "B":
                    Hdd += 2 * np.pi * bIS * (0.25 * (np.matmul(Sp[i], Sm[j]) + np.matmul(Sm[i], Sp[j])) * (3 * np.cos(theta) ** 2 - 1))
                elif mode == "C":
                    Hdd += 2 * np.pi * bIS * ((-3/2) * (np.matmul(Sp[i], Sz[j]) + np.matmul(Sz[i], Sp[j])) * np.sin(theta) * np.cos(theta) * np.exp(-1j * phi))
                elif mode == "D":
                    Hdd += 2 * np.pi * bIS * ((-3/2) * (np.matmul(Sm[i], Sz[j]) + np.matmul(Sz[i], Sm[j])) * np.sin(theta) * np.cos(theta) * np.exp(1j * phi))
                elif mode == "E":
                    Hdd += 2 * np.pi * bIS * ((-3/4) * np.matmul(Sp[i], Sp[j]) * np.sin(theta) ** 2 * np.exp(-1j * 2 * phi))
                elif mode == "F":
                    Hdd += 2 * np.pi * bIS * ((-3/4) * np.matmul(Sm[i], Sm[j]) * np.sin(theta) ** 2 * np.exp(1j * 2 * phi))

        return QunObj(Hdd)

    def Interaction_Hamiltonian_Catesian_Wigner(self, XQ, ApafQ, YQ, alpha=0.0, beta=0.0, gamma=0.0):
        """
        General interaction Hamiltonian using Wigner rotation of Cartesian tensors.

        Parameters:
        -----------
        XQ : str
            Name of the spin operator prefix (e.g., "I" or "S").
        ApafQ : QunObj
            Interaction tensor in PAF (Principal Axis Frame).
        YQ : str
            Second spin operator prefix (e.g., "S") or "" if interacting with external field.
        alpha, beta, gamma : float
            Euler angles in radians to rotate from PAF to lab frame.

        Returns:
        --------
        QunObj
            The interaction Hamiltonian.
        """
        X = [getattr(self.class_QS, XQ + "x").data, getattr(self.class_QS, XQ + "y").data, getattr(self.class_QS, XQ + "z").data]

        if YQ == "":
            if self.InteractioTensor_AngularFrequency: # Isotropic and Anisotropy are in angular frequency units
                Y = [
                    0.0 * np.eye(self.class_QS.Vdim),
                    0.0 * np.eye(self.class_QS.Vdim),
                    np.eye(self.class_QS.Vdim)
                ]
            else:
                Y = [
                    0.0 * np.eye(self.class_QS.Vdim),
                    0.0 * np.eye(self.class_QS.Vdim),
                    getattr(self.class_QS, XQ).gamma * self.class_QS.B0 * np.eye(self.class_QS.Vdim)
                ]

        else:
            Y = [getattr(self.class_QS, YQ + "x").data, getattr(self.class_QS, YQ + "y").data, getattr(self.class_QS, YQ + "z").data]

        # Spherical tensor decomposition
        Sptensor = ST.MatrixToSphericalTensors(ApafQ)

        T0 = Sptensor["rank0"]
        T11, T10, T1m1 = Sptensor["rank1"]
        T22, T21, T20, T2m1, T2m2 = Sptensor["rank2"]

        # Wigner D-matrix rotation
        Wigner_rank1 = PyOR_Rotation.Wigner_D_Matrix(1, alpha, beta, gamma)
        Wigner_rank2 = PyOR_Rotation.Wigner_D_Matrix(2, alpha, beta, gamma)

        Rot_rank1 = Wigner_rank1.data @ np.array([[T11], [T10], [T1m1]])
        Rot_rank2 = Wigner_rank2.data @ np.array([[T22], [T21], [T20], [T2m1], [T2m2]])

        Sptensor_f = {
            "rank0": T0,
            "rank1": [Rot_rank1[0], Rot_rank1[1], Rot_rank1[2]],
            "rank2": [Rot_rank2[0], Rot_rank2[1], Rot_rank2[2], Rot_rank2[3], Rot_rank2[4]]
        }

        AQ = ST.SphericalTensorsToMatrix(Sptensor_f)
        A = AQ.data

        return QunObj(
            A[0, 0] * X[0] @ Y[0] + A[1, 0] * X[1] @ Y[0] + A[2, 0] * X[2] @ Y[0] +
            A[0, 1] * X[0] @ Y[1] + A[1, 1] * X[1] @ Y[1] + A[2, 1] * X[2] @ Y[1] +
            A[0, 2] * X[0] @ Y[2] + A[1, 2] * X[1] @ Y[2] + A[2, 2] * X[2] @ Y[2]
        )

    def Interaction_Hamiltonian_Catesian_Euler(self, XQ, AQ, YQ, alpha=0.0, beta=0.0, gamma=0.0):
        """
        Rotate interaction tensor using Euler angles and construct Hamiltonian in Cartesian space.

        Parameters:
        -----------
        XQ : str
            Name of spin operator (e.g., "I")
        AQ : QunObj
            Interaction tensor in PAF
        YQ : str
            Second spin operator (e.g., "S"), or "" if external field
        alpha, beta, gamma : float
            Euler angles in radians

        Returns:
        --------
        QunObj
            The interaction Hamiltonian
        """
        X = [getattr(self.class_QS, XQ + "x").data, getattr(self.class_QS, XQ + "y").data, getattr(self.class_QS, XQ + "z").data]

        if YQ == "":
            if self.InteractioTensor_AngularFrequency: # Isotropic and Anisotropy are in angular frequency units
                Y = [
                    0.0 * np.eye(self.class_QS.Vdim),
                    0.0 * np.eye(self.class_QS.Vdim),
                    np.eye(self.class_QS.Vdim)
                ]
            else:
                Y = [
                    0.0 * np.eye(self.class_QS.Vdim),
                    0.0 * np.eye(self.class_QS.Vdim),
                    getattr(self.class_QS, XQ).gamma * self.class_QS.B0 * np.eye(self.class_QS.Vdim)
                ]
                                
        else:
            Y = [getattr(self.class_QS, YQ + "x").data, getattr(self.class_QS, YQ + "y").data, getattr(self.class_QS, YQ + "z").data]

        Atemp = AQ.data
        Rot = PyOR_Rotation.RotateEuler(alpha, beta, gamma)
        A = Rot @ Atemp @ Rot.T

        return QunObj(
            A[0, 0] * X[0] @ Y[0] + A[1, 0] * X[1] @ Y[0] + A[2, 0] * X[2] @ Y[0] +
            A[0, 1] * X[0] @ Y[1] + A[1, 1] * X[1] @ Y[1] + A[2, 1] * X[2] @ Y[1] +
            A[0, 2] * X[0] @ Y[2] + A[1, 2] * X[1] @ Y[2] + A[2, 2] * X[2] @ Y[2]
        )

    def Interaction_Hamiltonian_SphericalTensor(self, X, ApafQ, Y, string, approx, alpha=0.0, beta=0.0, gamma=0.0):
        """
        General Hamiltonian using spherical tensor formalism.

        Reference:
        ---------
        Michael Mehring, Internal Spin Interactions and Rotations in Solids.
        """

        # Transform tensor to lab frame via Wigner rotation
        Sptensor = ST.MatrixToSphericalTensors(ApafQ)
        AA0 = Sptensor["rank0"]
        AA11, AA10, AA1m1 = Sptensor["rank1"]
        AA22, AA21, AA20, AA2m1, AA2m2 = Sptensor["rank2"]

        Wigner_rank1 = PyOR_Rotation.Wigner_D_Matrix(1, -alpha, beta, gamma) # Note the negative sign !!!
        Wigner_rank2 = PyOR_Rotation.Wigner_D_Matrix(2, -alpha, beta, gamma)

        Rot_rank1 = Wigner_rank1.data @ np.array([[AA11], [AA10], [AA1m1]])
        Rot_rank2 = Wigner_rank2.data @ np.array([[AA22], [AA21], [AA20], [AA2m1], [AA2m2]])

        # Apply gyromagnetic ratio scaling for spin-field
        if string == "spin-field":
            if self.InteractioTensor_AngularFrequency: # Isotropic and Anisotropy are in angular frequency units
                A0 = AA0
                A11, A10, A1m1 = Rot_rank1[0], Rot_rank1[1], Rot_rank1[2]
                A22, A21, A20, A2m1, A2m2 = [x for x in Rot_rank2]

                Im, Ip, Iz = getattr(self.class_QS, X + "m").data, getattr(self.class_QS, X + "p").data, getattr(self.class_QS, X + "z").data

                # Tensor operators for spin-field
                T0 = (-1.0/np.sqrt(3)) * Iz
                T10 = 0.0 * Iz
                T11 = -0.5 * Ip
                T1m1 = -0.5 * Im
                T20 = (2.0/np.sqrt(6)) * Iz
                T21 = -0.5 * Ip
                T2m1 = 0.5 * Im
                T22 = 0.0 * Iz
                T2m2 = 0.0 * Iz
            else:
                gamma = getattr(self.class_QS, X).gamma
                A0 = gamma * AA0
                A11, A10, A1m1 = gamma * Rot_rank1[0], gamma * Rot_rank1[1], gamma * Rot_rank1[2]
                A22, A21, A20, A2m1, A2m2 = [gamma * x for x in Rot_rank2]

                Im, Ip, Iz = getattr(self.class_QS, X + "m").data, getattr(self.class_QS, X + "p").data, getattr(self.class_QS, X + "z").data
                B0 = self.class_QS.B0

                # Tensor operators for spin-field
                T0 = (-1.0/np.sqrt(3)) * Iz * B0
                T10 = 0.0 * Iz
                T11 = -0.5 * Ip * B0
                T1m1 = -0.5 * Im * B0
                T20 = (2.0/np.sqrt(6)) * Iz * B0
                T21 = -0.5 * Ip * B0
                T2m1 = 0.5 * Im * B0
                T22 = 0.0 * Iz
                T2m2 = 0.0 * Iz

        elif string == "spin-spin":
            A0 = AA0
            A11, A10, A1m1 = Rot_rank1[0], Rot_rank1[1], Rot_rank1[2]
            A22, A21, A20, A2m1, A2m2 = Rot_rank2

            Im, Ip, Iz = getattr(self.class_QS, X + "m").data, getattr(self.class_QS, X + "p").data, getattr(self.class_QS, X + "z").data
            Sm, Sp, Sz = getattr(self.class_QS, Y + "m").data, getattr(self.class_QS, Y + "p").data, getattr(self.class_QS, Y + "z").data

            Ix, Iy = 0.5 * (Ip + Im), -0.5j * (Ip - Im)
            Sx, Sy = 0.5 * (Sp + Sm), -0.5j * (Sp - Sm)

            # Tensor products for spin-spin interaction
            T0 = (-1/np.sqrt(3)) * (Ix @ Sx + Iy @ Sy + Iz @ Sz)
            T10 = (-0.5/np.sqrt(2)) * (Ip @ Sm - Im @ Sp)
            T11 = 0.5 * (Iz @ Sp - Ip @ Sz)
            T1m1 = 0.5 * (Iz @ Sm - Im @ Sz)
            T20 = (1/np.sqrt(6)) * (3 * Iz @ Sz - (Ix @ Sx + Iy @ Sy + Iz @ Sz))
            T21 = -0.5 * (Iz @ Sp + Ip @ Sz)
            T2m1 = 0.5 * (Iz @ Sm + Im @ Sz)
            T22 = 0.5 * Ip @ Sp
            T2m2 = 0.5 * Im @ Sm

        # Final assembly
        if approx == "all":
            return QunObj(
                A0 * T0 + A10 * T10 - (A11 * T1m1 + A1m1 * T11) +
                A20 * T20 - (A21 * T2m1 + A2m1 * T21) + A22 * T2m2 + A2m2 * T22
            )
        if approx == "secular":
            return QunObj(np.diag(np.diag(
                A0 * T0 + A10 * T10 - (A11 * T1m1 + A1m1 * T11) +
                A20 * T20 - (A21 * T2m1 + A2m1 * T21) + A22 * T2m2 + A2m2 * T22
            )))
        if approx == "secular + pseudosecular":
            return QunObj(A0 * T0 + A10 * T10 + A20 * T20)

    def Interaction_Hamiltonian_LAB_CSA_Secular(self, X, ApasQ, theta, phi):
        """
        Constructs the secular CSA Hamiltonian in the lab frame.

        Parameters:
        -----------
        X : str
            Spin label (e.g., 'I' or 'S')
        ApasQ : QunObj
            CSA tensor in PAF (principal axis frame)
        theta : float
            Polar angle in degrees
        phi : float
            Azimuthal angle in degrees

        Returns:
        --------
        QunObj
            The secular CSA Hamiltonian component.
        """
        Apas = ApasQ.data
        gamma = getattr(self.class_QS, X).gamma
        SZ = getattr(self.class_QS, X + "z").data

        theta = (np.pi / 180.0) * theta
        phi = (np.pi / 180.0) * phi

        values = self.InteractionTensor_PAF_Decomposition(ApasQ)
        Isotropic = values["Isotropic"] * 2.0 * np.pi
        Anisotropy = values["Anisotropy"] * 2.0 * np.pi
        Asymmetry = values["Asymmetry"]

        rho_lab_zz = Isotropic + 0.5 * Anisotropy * (
            3.0 * (np.cos(theta))**2 - 1 - Asymmetry * ((np.sin(theta))**2 * np.cos(2.0 * phi))
        )

        if self.InteractioTensor_AngularFrequency: # Isotropic and Anisotropy are in angular frequency units
            return QunObj(rho_lab_zz  * SZ)
        else:
            return QunObj(gamma * rho_lab_zz * self.class_QS.B0 * SZ)

    def Interaction_Hamiltonian_LAB_Quadrupole_Secular(self, X, Coupling, etaQ, theta, phi):
        """
        Constructs the secular quadrupole Hamiltonian.

        Parameters:
        -----------
        X : str
            Nucleus label
        eq : float
            Electric field gradient
        etaQ : float
            Quadrupolar asymmetry parameter
        theta, phi : float
            Orientation angles in degrees

        Returns:
        --------
        QunObj
            The quadrupole Hamiltonian
        """
        
        Coupling = 2 * np.pi * Coupling
        spin = getattr(self.class_QS, X).spin

        constant = Coupling / (4.0 * spin * (2.0 * spin - 1))

        Sx = getattr(self.class_QS, X + "x").data
        Sy = getattr(self.class_QS, X + "y").data
        Sz = getattr(self.class_QS, X + "z").data

        theta = (np.pi / 180.0) * theta
        phi = (np.pi / 180.0) * phi

        return QunObj(
            constant * (3.0 * Sz @ Sz - (Sx @ Sx + Sy @ Sy + Sz @ Sz)) *
            0.5 * (3.0 * (np.cos(theta))**2 - 1 - etaQ * (np.sin(theta))**2 * np.cos(2.0 * phi))
        )

    def InteractionTensor_LAB(self, ApafQ, alpha=0.0, beta=0.0, gamma=0.0):
        """
        Rotates a tensor from PAF to LAB frame using spherical tensors and Wigner rotation.

        Parameters:
        -----------
        ApafQ : QunObj
            Tensor in principal axis frame
        alpha, beta, gamma : float
            Euler angles

        Returns:
        --------
        QunObj
            Rotated tensor in LAB frame
        """
        Sptensor = ST.MatrixToSphericalTensors(ApafQ)
        T0 = Sptensor["rank0"]
        T11, T10, T1m1 = Sptensor["rank1"]
        T22, T21, T20, T2m1, T2m2 = Sptensor["rank2"]

        Wigner_rank1 = PyOR_Rotation.Wigner_D_Matrix(1, alpha, beta, gamma)
        Wigner_rank2 = PyOR_Rotation.Wigner_D_Matrix(2, alpha, beta, gamma)

        Rot_rank1 = Wigner_rank1.data @ np.array([[T11], [T10], [T1m1]])
        Rot_rank2 = Wigner_rank2.data @ np.array([[T22], [T21], [T20], [T2m1], [T2m2]])

        Sptensor_f = {
            "rank0": T0,
            "rank1": [Rot_rank1[0], Rot_rank1[1], Rot_rank1[2]],
            "rank2": [Rot_rank2[0], Rot_rank2[1], Rot_rank2[2], Rot_rank2[3], Rot_rank2[4]]
        }

        AQ = ST.SphericalTensorsToMatrix(Sptensor_f)
        return AQ

    def InteractionTensor_PAF_CSA(self, Iso, Aniso, Asymmetry):
        """
        Constructs the CSA interaction tensor in the principal axis frame (PAF).

        Parameters:
        -----------
        Iso : float
            Isotropic component
        Aniso : float
            Anisotropy (Azz - Iso)
        Asymmetry : float
            Asymmetry parameter (eta = (Ayy - Axx) / Aniso),  range from 0 <= eta <= 1

        Returns:
        --------
        QunObj
            3x3 CSA tensor in PAF
        """

        if self.InteractioTensor_AngularFrequency: # Isotropic and Anisotropy are in angular frequency units
            Iso = 2.0 * np.pi * Iso
            Aniso = 2.0 * np.pi * Aniso

        I1 = Iso * np.eye(3)
        I2 = np.eye(3)
        I2[0][0] = -0.5 * (1 + Asymmetry)
        I2[1][1] = -0.5 * (1 - Asymmetry)

        return QunObj(I1 + Aniso * I2)

    def InteractionTensor_PAF_Quadrupole(self, X, Coupling, etaQ):
        """
        Constructs the quadrupolar tensor in the PAF.

        Parameters:
        -----------
        X : str
            Spin label
        eq : float
            Electric field gradient
        etaQ : float
            Quadrupolar asymmetry parameter

        Returns:
        --------
        QunObj
            Quadrupolar tensor in the PAF
        """
        
        Coupling = 2.0 * np.pi * Coupling
        spin = getattr(self.class_QS, X).spin

        constant = Coupling / (2.0 * spin * (2.0 * spin - 1))

        I = np.eye(3)
        I[0][0] = -0.5 * (1 + etaQ) * constant
        I[1][1] = -0.5 * (1 - etaQ) * constant
        I[2][2] = constant

        return QunObj(I)

    def InteractionTensor_PAF_Dipole(self, d):
        """
        Constructs a PAF dipolar tensor.

        Parameters:
        -----------
        d : float
            Dipolar coupling constant (in Hz)

        Returns:
        --------
        QunObj
            Dipolar tensor in the PAF
        """
        d = 2.0 * np.pi * d

        I = np.eye(3)
        I[0][0] = d
        I[1][1] = d
        I[2][2] = -2 * d

        return QunObj(I)

    def InteractionTensor_PAF_Decomposition(self, AQ):
        """
        Decomposes a PAF tensor into isotropic, anisotropy, and asymmetry components.

        Parameters:
        -----------
        AQ : QunObj
            Tensor to decompose

        Returns:
        --------
        dict
            Dictionary with keys: "Isotropic", "Anisotropy", "Asymmetry"
        """
        A = AQ.data
        output = {}

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")

        trace_A = np.trace(A)
        A_iso = trace_A / 3
        output["Isotropic"] = np.real(A_iso / (2.0 * np.pi))

        aniso = A[2][2] - A_iso
        output["Anisotropy"] = np.real(aniso / (2.0 * np.pi))

        asymm = (A[1][1] - A[0][0]) / (A[2][2] - A_iso)
        output["Asymmetry"] = np.real(asymm)

        return output

    def InteractionTensor_LAB_Decomposition(self, AQ):
        """
        Decomposes a LAB-frame tensor into isotropic, symmetric, and antisymmetric parts.

        Parameters:
        -----------
        AQ : QunObj
            LAB-frame tensor

        Returns:
        --------
        dict
            Dictionary with QunObj entries:
            "Isotropic", "Symmetric", "Antisymmetric"
        """
        A = AQ.data

        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")

        output = {}
        trace_A = np.trace(A)
        A_iso = (trace_A / 3) * np.eye(A.shape[0])
        output["Isotropic"] = QunObj(A_iso)

        A_sym = 0.5 * (A + A.T)
        output["Symmetric"] = QunObj(A_sym)

        A_asym = 0.5 * (A - A.T)
        output["Antisymmetric"] = QunObj(A_asym)

        return output

    def PowderSpectrum(self, EVol, rhoI, rhoeq, X, IT_PAF, Y, string, approx,
                    alpha, beta, gamma, weighted=True, weight=None,
                    SecularEquation="spherical", ncores = -1):
        """
        Computes the powder-averaged spectrum over (alpha, beta, gamma) angles.

        Parameters:
        -----------
        weighted : bool
            Whether to use weighted averaging.
        weight : ndarray or None
            Optional crystallite weights. If None and weighted=True,
            defaults to sin(beta) weighting.

        Returns:
        --------
        freq : ndarray
            Frequency axis.
        spectrum : ndarray
            Powder-averaged spectrum (absolute value).
        """

        alpha_beta_gamma_pairs = list(zip(alpha, beta, gamma))

        def compute_single(alpha_i, beta_i, gamma_i):
            if SecularEquation == "spherical":
                HAM = self.Interaction_Hamiltonian_SphericalTensor(
                    X, IT_PAF, Y, string, approx, alpha_i, beta_i, gamma_i
                )
            elif SecularEquation == "csa":
                HAM = self.Interaction_Hamiltonian_LAB_CSA_Secular(
                    X, IT_PAF, beta_i, alpha_i
                )
            else:
                raise ValueError(f"Unknown SecularEquation: {SecularEquation}")

            t, rho_t = EVol.Evolution(rhoI, rhoeq, HAM)

            if Y == "":
                det_Mt = getattr(self.class_QS, X + "p").data
            else:
                det_Mt = getattr(self.class_QS, X + "p").data + getattr(self.class_QS, Y + "p").data

            t, Mt = EVol.Expectation(rho_t, det_Mt)
            Mt = Spro.WindowFunction(t, Mt, 0.5)
            freq, spectrum_single = Spro.FourierTransform(Mt, self.class_QS.AcqFS, 5)
            return freq, np.abs(spectrum_single)

        # Run all computations in parallel
        results = Parallel(n_jobs=ncores)(
            delayed(compute_single)(alpha_i, beta_i, gamma_i)
            for alpha_i, beta_i, gamma_i in alpha_beta_gamma_pairs
        )

        freq_list, spectra = zip(*results)
        freq = freq_list[0]  # assume all same

        if weighted:
            if weight is not None:
                # Use external crystallite weight array
                weights = np.asarray(weight)
            else:
                # Use sin(beta) weighting
                weights = np.sin(np.radians(beta))
            weights = weights / np.sum(weights)  # Normalize
            spectrum = np.sum([w * s for w, s in zip(weights, spectra)], axis=0)
        else:
            # Uniform averaging (not normalized)
            spectrum = np.sum(spectra, axis=0)

        return freq, spectrum

    def ShapedPulse_Bruker(self, file_path, pulseLength, RotationAngle):
        """
        Load and process a shaped pulse from a Bruker shape file.

        Parameters:
        -----------
        file_path : str
            Path to the shaped pulse file (typically Bruker format)
        pulseLength : float
            Total length of the shaped pulse (in seconds)
        RotationAngle : float
            Desired rotation angle (in degrees)

        Returns:
        --------
        tuple
            (time array, B1 amplitude over time, B1 phase over time)

        References:
        -----------
        - Bruker Shape Tool manual
        - https://github.com/modernscientist/...
        """
        nu1_hard_pulseLength = RotationAngle / (360.0 * pulseLength)
        print("Nutation frequency of hard pulse (Hz):", nu1_hard_pulseLength)

        with open(file_path, 'r') as f:
            pulseString = f.read()

        pulseShapeArray = np.genfromtxt(StringIO(pulseString), comments='#', delimiter=',')
        n_pulse = pulseShapeArray.shape[0]

        pulseShapeInten = pulseShapeArray[:, 0] / np.max(np.abs(pulseShapeArray[:, 0]))
        pulseShapePhase = pulseShapeArray[:, 1] * np.pi / 180

        xPulseShape = pulseShapeInten * np.cos(pulseShapePhase)
        yPulseShape = pulseShapeInten * np.sin(pulseShapePhase)
        scalingFactor = np.sum(xPulseShape) / n_pulse

        print("Scaling Factor:", scalingFactor)

        nuB1max = nu1_hard_pulseLength / scalingFactor

        print("Maximum nuB1 (Hz):", nuB1max)
        print("Period corresponding to maximum nuB1 (s):", 1.0 / nuB1max)

        time = np.linspace(0, pulseLength, n_pulse)

        return time, nuB1max * pulseShapeInten, pulseShapePhase

    def ShapedPulse_Interpolate(self, time, SPIntensity, SPPhase, Kind):
        """
        Interpolate amplitude and phase arrays of a shaped pulse.

        Parameters:
        -----------
        time : array
            Time axis of the pulse shape
        SPIntensity : array
            Amplitude values of the pulse shape
        SPPhase : array
            Phase values of the pulse shape (radians)
        Kind : str
            Interpolation method (e.g., 'linear', 'cubic', etc.)

        Returns:
        --------
        tuple
            (amplitude interpolator, phase interpolator)
        """
        return (
            interp1d(time, SPIntensity, kind=Kind, fill_value="extrapolate"),
            interp1d(time, SPPhase, kind=Kind, fill_value="extrapolate")
        )

    def Eigen(self, H):
        """
        Compute eigenvalues and eigenvectors of a Hamiltonian.

        Parameters:
        -----------
        H : np.ndarray
            Hamiltonian matrix

        Returns:
        --------
        tuple
            (eigenvalues, eigenvectors)
        """
        eigenvalues, eigenvectors = lina.eig(H)
        return eigenvalues, eigenvectors



