"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain class Hamiltonian
"""

import numpy as np
import PhysicalConstants 
import Rotation
from QuantumObject import QunObj

class Hamiltonian:
    def __init__(self, class_QS):
        self.class_QS = class_QS
        self.hbar = PhysicalConstants.constants("hbar")
        self.mu0 = PhysicalConstants.constants("mu0")
        self.LarmorF = self.LarmorFrequency()
        
        # Inverse of 2 Pi
        self.Inverse2PI = 1.0/(2.0 * np.pi)        
      
    def LarmorFrequency(self):
            """
            Generate Larmor Frequency, Omega0 in Lab Frame
            
            INPUT
            -----
            Gamma: List of Gyromagnetic ratios of individual spins
            B0: Field of the spectrometer in Tesla
            Offset: List of the chemical shifts of individual spins
            
            OUTPUT
            ------
            return array of Larmor frequencies of individual spins in lab frame
            """
            Gamma = self.class_QS.Gamma
            B0 = self.class_QS.B0
            Offset = self.class_QS.Offset
            
            W0 = np.zeros((self.class_QS.Nspins))
            gamma = np.asarray(Gamma)
            offset = np.asarray(Offset)
            for i in range(self.class_QS.Nspins):
                W0[i] = -1 * gamma[i] * B0 - 2 * np.pi * offset[i]
            
            if self.class_QS.print_Larmor:
                print("Larmor Frequency in MHz: ", W0/2.0/np.pi/1.0e6)    
            return W0    
        
    def Zeeman(self):
        """
        Generating Zeeman Hamiltonian in Lab Frame
        
        INPUT
        ----
        LarmorF: Array of Larmor frequencies of individual spins in lab frame (LarmorF = System.LarmorFrequency(Gamma,B0,Offset))
        Sz: Sz spin operators
        
        OUTPUT
        ------
        HZ: Zeeman hamiltonian in lab Frame (Angluar frequency Units) 
        """

        LarmorF = self.LarmorF
        Sz = self.class_QS.Sz_
        
        Hz = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        for i in range(self.class_QS.Nspins):
            Hz = Hz + LarmorF[i] * Sz[i]
                
        return QunObj(Hz)

    def Zeeman_RotFrame(self):
        """
        Generating Zeeman Hamiltonian in Rotating Frame
        
        INPUTS
        ------
        LarmorF: Array of Larmor frequencies of individual spins in lab frame (LarmorF = System.LarmorFrequency(Gamma,B0,Offset))
        Sz: Sz spin operators
        OmegaRF: List of rotating frame frequencies 
                 Homonuclear case - All frequencies are the same
                 Hetronuclear case - ??

        OUTPUT
        ------
        HZ: Zeeman hamiltonian in rotating Frame  (Angluar frequency Units)       
        """

        LarmorF = self.LarmorF
        OmegaRF = self.class_QS.OmegaRF
        Sz = self.class_QS.Sz_
        
        omegaRF = np.asarray(OmegaRF)
        Hz = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        for i in range(self.class_QS.Nspins):
            Hz = Hz + (LarmorF[i]-omegaRF[i]) * Sz[i]
        return QunObj(Hz)
                
    def Zeeman_B1(self,Omega1,Omega1Phase):  
        """
        Generating Zeeman Hamiltonian with B1 Hamiltonian (Time independent Hamiltonian)
        
        INPUT
        -----
        Sx: Sx spin operators
        Sy: Sy spin operators
        Omega1: Amplitude of RF signal in Hz (nutation frequency)
        Omega1Phase: Phase of RF signal in deg
                
        OUTPUT
        ------
        HzB1: B1 field hamiltonian (Angluar frequency Units) 
        """ 
        
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        
        HzB1 = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        omega1 = 2*np.pi*Omega1
        Omega1Phase = np.pi*Omega1Phase/180.0
        for i in range(self.class_QS.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1Phase) + Sy[i]*np.sin(Omega1Phase))
        return QunObj(HzB1)
        
    def Zeeman_B1_Offresonance(self,t,Omega1,Omega1freq,Omega1Phase):  
        """
        Generating Zeeman Hamiltonian with B1 Hamiltonian - Off resonance (Time dependent Hamiltonian)
        
        INPUT
        -----
        t: time
        Sx: Sx spin operators
        Sy: Sy spin operators
        Omega1: Amplitude of RF signal in Hz (nutation frequency)
        Omega1freq: RF frequency in rotating frame (Hz)
        Omega1Phase: Phase of RF signal in deg
                
        OUTPUT
        ------
        HzB1: B1 field hamiltonian (Angluar frequency Units) 
        """  

        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        
        HzB1 = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        omega1 = 2*np.pi*Omega1
        Omega1freq = 2*np.pi*Omega1freq
        Omega1Phase = np.pi*Omega1Phase/180.0
        for i in range(self.class_QS.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1freq * t + Omega1Phase) + Sy[i]*np.sin(Omega1freq * t + Omega1Phase))
        return QunObj(HzB1)        

    def Zeeman_B1_ShapedPulse(self,t,Omega1T,Omega1freq,Omega1PhaseT):  
        """
        Generating Zeeman Hamiltonian with B1 Hamiltonian - Gaussian (Time dependent Hamiltonian)
        
        INPUT
        -----
        t: time
        Sx: Sx spin operators
        Sy: Sy spin operators
        Omega1: Amplitude of RF signal in Hz (nutation frequency)
        Omega1freq: RF frequency in rotating frame (Hz)
        Omega1Phase: Phase of RF signal in deg
                
        OUTPUT
        ------
        HzB1: B1 field hamiltonian (Angluar frequency Units) 
        """  

        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        
        HzB1 = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)

        omega1 = 2*np.pi*(Omega1T(t))
        Omega1freq = 2*np.pi*Omega1freq
        Omega1Phase = (Omega1PhaseT(t))
        for i in range(self.class_QS.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1freq * t + Omega1Phase) + Sy[i]*np.sin(Omega1freq * t + Omega1Phase))                  
        return QunObj(HzB1) 
        
    def Jcoupling(self):    
        """
        Generate J coupling Hamiltonian    
        
        INPUT
        -----
        J: J coupling constant (Hz), J[ i ][ j ], J coupling between 'i' th spin and 'j' th spin and j>i and j != i
        Sx: Sx spin operators
        Sy: Sy spin operators
        Sz: Sz spin operators 
        
        OUTPUT
        ------
        Hj: J coupling Hamiltonian (Angluar frequency Units) 
        """ 
        
        J = self.class_QS.Jlist
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        Sz = self.class_QS.Sz_
        
        J = np.triu(2*np.pi*J)    
        Hj = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        for i in range(self.class_QS.Nspins):
            for j in range(self.class_QS.Nspins):
                Hj = Hj + J[i][j] * (np.matmul(Sx[i],Sx[j]) + np.matmul(Sy[i],Sy[j]) + np.matmul(Sz[i],Sz[j]))      
        return QunObj(Hj)        

    def Jcoupling_Weak(self):    
        """
        Generate J coupling Hamiltonian (weak)

        INPUT
        -----
        J: J coupling constant (Hz)
        Sz: Sz spin operators 
        
        OUTPUT
        ------
        Hj: J coupling Hamiltonian (weak)  (Angluar frequency Units)             
        """ 
        
        J = self.class_QS.Jlist
        Sz = self.class_QS.Sz_
        
        J = np.triu(2*np.pi*J)    
        Hj = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        for i in range(self.class_QS.Nspins):
            for j in range(self.class_QS.Nspins):
                Hj = Hj + J[i][j] * np.matmul(Sz[i],Sz[j])      
        return QunObj(Hj) 
        
    def Dipole_Coupling_Constant(self,Gamma1,Gamma2,distance):
        """
        Dipolar coupling constant between two spins
        
        INPUT
        -----
        Gamma1: Gamma of Spin 1
        Gamma2: Gamma of Spin 2
        distance: distance between spin 1 and 2
        
        OUTPUT
        ------
        return dipolar coupling constant (in Hz)
        """   
        print("dipolar coupling constant (in Hz)")
        return  self.mu0 * Gamma1 * Gamma2 * self.hbar * (distance**-3) / (4 * np.pi) / (2*np.pi)
                
    def DDcoupling(self):
        """
        Generate Dipole-Dipole coupling Hamiltonian
        INPUT
        -----
        Sx,Sy,Sz,S+ and S-: Spin operators
        spin_index: index of two psins, list
        bIS : - mu0 * Gamma(I) * Gamma(S) * hbar /(4 * PI * (rIS)**3)
        theta: polar angle, angle between z and position vector between two psin
        phi: azimuthal angle, angle between x and project of position vector on xy plane
        secular: If True secular approximation is used
                 
        OUTPUT     
        ------
        Hdd : Dipole-Dipole coupling Hamitonian of the system (Angluar frequency Units)         
        """ 

        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        Sz = self.class_QS.Sz_
        Sp = self.class_QS.Sp_
        Sm = self.class_QS.Sm_
        
        thetaAll, phiAll = np.array(self.class_QS.DipoleAngle).T
        thetaAll = np.pi*thetaAll/180.0
        phiAll = np.pi*phiAll/180.0
        
        Spin1, Spin2 = np.array(self.class_QS.DipolePairs).T
        
        Hdd = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim),dtype=np.double)
        for i,j,string,theta,phi,bIS in zip(Spin1, Spin2, self.class_QS.Dipole_ZeemanTruncation,thetaAll,phiAll,self.class_QS.DipolebIS):
        
            if string == "secular Hetronuclear":
                A = np.matmul(Sz[i],Sz[j]) * (3 * (np.cos(theta))**2 - 1)
                Hdd = Hdd + 2.0*np.pi * bIS * A

            if string == "secular Homonuclear":
                A = np.matmul(Sz[i],Sz[j]) * (3 * (np.cos(theta))**2 - 1)
                B = (-1/4) * (np.matmul(Sp[i],Sm[j]) + np.matmul(Sm[i],Sp[j])) * (3 * (np.cos(theta))**2 - 1)
                Hdd = Hdd + 2.0*np.pi * bIS * (A + B)

            if string == "All":     
                A = np.matmul(Sz[i],Sz[j]) * (3 * (np.cos(theta))**2 - 1)
                B = (-1/4) * (np.matmul(Sp[i],Sm[j]) + np.matmul(Sm[i],Sp[j])) * (3 * (np.cos(theta))**2 - 1)
                C = (3/2) * (np.matmul(Sp[i],Sz[j]) + np.matmul(Sz[i],Sp[j])) * np.sin(theta) * np.cos(theta) * np.exp(-1j*phi)
                D = (3/2) * (np.matmul(Sm[i],Sz[j]) + np.matmul(Sz[i],Sm[j])) * np.sin(theta) * np.cos(theta) * np.exp(1j*phi)
                E = (3/4) * np.matmul(Sp[i],Sp[j]) * (np.sin(theta))**2 * np.exp(-1j * 2 * phi)
                F = (3/4) * np.matmul(Sm[i],Sm[j]) * (np.sin(theta))**2 * np.exp(1j * 2 * phi)
                Hdd = Hdd + 2.0*np.pi * bIS * (A+B+C+D+E+F)                
            
        return QunObj(Hdd) 
    
    def Hamiltonian_General(self, constant, X, Apas, Y, alpha=0.0, beta=0.0, gamma=0.0):
        """
        General expression of Hamiltonian
        """

        Rot = Rotation.RotateEuler(alpha, beta, gamma)
        A =  Rot @ Apas @ Rot.T

        return constant * (A[0, 0] * X[0] @ Y[0] + A[1, 0] * X[1] @ Y[0] + A[2, 0] * X[2] @ Y[0] +
                A[0, 1] * X[0] @ Y[1] + A[1, 1] * X[1] @ Y[1] + A[2, 1] * X[2] @ Y[1] +
                A[0, 2] * X[0] @ Y[2] + A[1, 2] * X[1] @ Y[2] + A[2, 2] * X[2] @ Y[2])

    def Hamiltonian_General_SphericalAngles(self, constant, X, Apas, Y, theta, phi):
        """
        General expression of Hamiltonian
        """
        Rot = Rotation.RotateX(theta) @ Rotation.RotateZ(phi)
        A =  Rot @ Apas @ Rot.T

        return constant * (A[0, 0] * X[0] @ Y[0] + A[1, 0] * X[1] @ Y[0] + A[2, 0] * X[2] @ Y[0] +
                A[0, 1] * X[0] @ Y[1] + A[1, 1] * X[1] @ Y[1] + A[2, 1] * X[2] @ Y[1] +
                A[0, 2] * X[0] @ Y[2] + A[1, 2] * X[1] @ Y[2] + A[2, 2] * X[2] @ Y[2])

    def Hamiltonian_CSA_ZeemanTruncation(self,gamma,SZ,Apas,theta,phi):
        """
        
        """

        theta = (np.pi/180.0) * theta
        phi = (np.pi/180.0) * phi
        omega = 2*np.pi*omega

        trace_A = np.trace(Apas)
        A_iso = (trace_A / 3)

        rho_lab_zz =  A_iso  - Apas[0][0] * (np.sin(theta))**2 * (np.cos(phi))**2 +  Apas[1][1] * (np.sin(theta))**2 * (np.sin(phi))**2  + Apas[2][2] * (np.cos(theta))**2

        return gamma * rho_lab_zz * self.B0 * SZ
    
    def InteractionTensor_PAS_CSA(self,Iso,Aniso,Asymmetry):
        """
        
        """
        I1 = Iso * np.eye(3)
        I2 = np.eye(3)
        I2[0][0] = -0.5 * (1 + Asymmetry)
        I2[1][1] = -0.5 * (1 - Asymmetry)
        return I1 + Aniso * I2

    def InteractionTensor_PAS_Dipole(self,d):
        """
        
        """
        I = np.eye(3)
        I[0][0] = d
        I[1][1] = d
        I[2][2] = -2 * d
        return I

    def InteractionTensor_PAS_Decomposition(self,A,string):
        """
        
        """

        # Ensure A is a square matrix
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")

        trace_A = np.trace(A)
        A_iso = (trace_A / 3)

        if string == "Isotropic":
            return A_iso

        if string == "Anisotropy":
            aniso = A[2][2] - A_iso 
            return aniso 

        if string == "Asymmetry":        
            asymm = (A[1][1] - A[0][0]) / (A[2][2] - A_iso )
            return asymm         

    def Matrix_Decomposition(self,A,string):
        """
        Splits the matrix A into its isotropic, symmetric, and antisymmetric parts.
        
        Arguments:
        A -- The input square matrix to split.
        
        Returns:
        A_iso -- Isotropic part of A.
        A_sym -- Symmetric part of A.
        A_asym -- Antisymmetric part of A.
        """
        
        # Ensure A is a square matrix
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")
        
        if string == "Isotropic":
            trace_A = np.trace(A)
            A_iso = (trace_A / 3) * np.eye(A.shape[0])
            return A_iso

        if string == "Symmetric":
            A_sym = 0.5 * (A + A.T)
            return A_sym 

        if string == "Antisymmetric":        
            A_asym = 0.5 * (A - A.T)
            return A_asym              