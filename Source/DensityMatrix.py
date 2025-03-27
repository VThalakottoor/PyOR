"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain class equlibrium density matrix
"""

import numpy as np
from scipy.linalg import expm
import PhysicalConstants 
import Rotation
from QuantumObject import QunObj

class DensityMatrix:
    def __init__(self, class_QS, class_Ham):
        self.class_QS = class_QS
        self.class_Ham = class_Ham
        self.hbar = PhysicalConstants.constants("hbar")
        self.mu0 = PhysicalConstants.constants("mu0")
        self.kb = PhysicalConstants.constants("kb")
  

    def EqulibriumDensityMatrix(self,Ti,HT_approx):
        """
        Equlibrium Density Matrix (Define individual spin temperature)
        
        INPUT
        -----
        LarmorF        : Larmor Frequency of each spins
        Sz             : Sz spin operators of each spins
        Ti             : Spin temperature of each spin
        
        OUTPUT     
        ------
        Equlibrium Density Matrix      
        """
        
        LarmorF = self.class_Ham.LarmorF
        Sz = self.class_QS.Sz_

        rho_T = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim))
        
        H_Eq_T = np.zeros((self.class_QS.Vdim,self.class_QS.Vdim))
        
        for i in range(self.class_QS.Nspins):
            H_Eq_T = H_Eq_T + self.class_QS.Convert_FreqUnitsTOEnergy(LarmorF[i] * Sz[i]) / (self.kb*Ti[i])  
         
        if HT_approx:
            E = np.eye(self.class_QS.Vdim)
            rho_T = (E - H_Eq_T)/np.trace(E - H_Eq_T) # High Temperature Approximation
        else:    
            rho_T = expm(-H_Eq_T)/np.trace(expm(-H_Eq_T)) # General
            
        print("Trace of density metrix = ", (np.trace(rho_T)).real)    

        return QunObj(rho_T) 
        
    def PolarizationVector(self,spinQ,rho,Sz,PolPercentage):
        """
        Polarization
        
        INPUT
        -----
        rho: density matrix
        Sz: Sz spin operator
        
        OUTPUT
        ------
        return polarization
        """ 
        if PolPercentage:   
            return 100 * (-(1.0/spinQ) * np.trace(np.matmul(rho,Sz))/np.trace(rho)).real
        else:   
            return (-(1.0/spinQ) * np.trace(np.matmul(rho,Sz))/np.trace(rho)).real 
