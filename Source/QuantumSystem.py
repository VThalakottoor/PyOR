"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.coml

This file contain class QuantumSystem

Attribute:
    ...

Methods:
    ...    

"""

# ---------- Package

from QuantumObject import QunObj
import PhysicalConstants 
import SpinQuantumNumber 
import Gamma
import QuadrupoleMoment 
import Particle

import numpy as np
from numpy import linalg as lina

import sympy as sp
from sympy import * # Remove ??
from sympy.physics.quantum.cg import CG

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import SpanSelector

import time

import scipy
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.linalg as la

import os
import sys
#sys.setrecursionlimit(1500)

import numba
from numba import njit, cfunc

from IPython.display import display, Latex, Math

from fractions import Fraction

import re
from io import StringIO

from functools import reduce

from collections import defaultdict

import math

# ---------- Package

class QuantumSystem:
    def __init__(self, SpinList,PrintDefault=True):

        #Spin List
        self.SpinList = SpinList

        # Store the dictionary keys as spin labels (e.g., ["A", "B", "C", ...])
        self.SpinDic = list(SpinList.keys())

        #Spin Indexing
        self.SpinIndex = {value: index for index, value in enumerate(self.SpinDic)}

        # Spin Name
        self.SpinName = np.array(list(SpinList.values()))

        # Spin values
        SPINLIST = []
        for i in self.SpinName:
            SPINLIST.append(SpinQuantumNumber.spin(i))

        # Store spin values as a array (e.g., [1/2, 1, 1/2])
        self.slist = np.array(SPINLIST)

        # Number of spins
        self.Nspins = len(self.slist)

        # Array of dimensions of individual Hilbert Space    
        Sdim = np.array([np.arange(-s, s + 1, 1).shape[-1] for s in self.slist])
        self.Sdim = Sdim

        # Dimension of Hilbert Space
        self.Vdim = np.prod(self.Sdim)

        # Dimension of Liouville Space
        self.Ldim = (self.Vdim) ** 2

        # Inverse of 2 Pi
        self.Inverse2PI = 1.0/(2.0 * np.pi)

        # Print default parameters
        if PrintDefault:
            print("\nPyOR default parameters/settings")
            print("--------------------------------")

        # Planck Constant
        self.hbarEQ1 = True
        if PrintDefault:
            print("\nDefine energy units: hbarEQ1 = ",self.hbarEQ1)

        # Matrix element tolarence
        self.MatrixTolarence = 1.0e-10
        if PrintDefault:
            print("\nDefine the matrix tolerence (make matrix elements less than tolarence value to zero): MatrixTolarence = ",self.MatrixTolarence)       

        # Gyromagnetic Ration     
        GAMMALIST = []
        for i in self.SpinName:
            GAMMALIST.append(Gamma.gamma(i))
        self.Gamma = GAMMALIST
        if PrintDefault:
            print("\nDefine the gyromagnetic ratios: Gamma = ",self.Gamma)

        # Spectrometer Field in T
        self.B0 = None
        if PrintDefault:
            print("\nDefine the static field along Z: B0 = ",self.B0)
        
        # Rotating Frame Frequency
        self.OmegaRF = [0] * self.Nspins 
        if PrintDefault:
            print("\nDefine rotating frame frequency: OmegaRF = ",self.OmegaRF)
        
        # Offset Frequency in rotating frame (Hz) 
        self.Offset = [0] * self.Nspins  
        if PrintDefault:
            print("\nDefine the offset frequencies of the spins: Offset = ",self.Offset) 

        # Larmor Frequency
        self.print_Larmor = True
        if PrintDefault:
            print("\nDo you want to print the larmor frequency: print_Larmor = ",self.print_Larmor)        
        
        # J Coupling 
        self.Jlist = np.zeros((self.Nspins,self.Nspins))
        if PrintDefault:
            print("\nDefine the J coupling: Jlist = \n",self.Jlist)
        
        # Dipole dipole pairs tuple of list, spin pair interact by dipolar coupling, useful for relaxation  
        self.DipolePairs = []
        if PrintDefault:
            print("\nDefine the spin paris dipolar coupled: DipolePairs = ",self.DipolePairs)
        self.Dipole_ZeemanTruncation = []
        if PrintDefault:
            print("\nDefine the spin paris Zeeman truncation: Dipole_ZeemanTruncation = ",self.Dipole_ZeemanTruncation)
        self.DipoleAngle = []
        if PrintDefault:
            print("\nDefine the spin paris dipole angle (theta,phi): DipoleAngle = ",self.DipoleAngle)
        self.DipolebIS = []
        if PrintDefault:
            print("\nDefine the spin paris coupling constants: DipolebIS = ",self.DipolebIS)

        # Initial and final spin temperature
        if PrintDefault:    
            print("\nSpin temperatures")
            print("----------------")
        self.Ispintemp = [0] * self.Nspins
        if PrintDefault:
            print("Initial spin temperature of individual spins: Ispintemp = ",self.Ispintemp)
        self.Fspintemp = [0] * self.Nspins
        if PrintDefault:
            print("Final spin temperature of individual spins = ",self.Fspintemp)

        # Propagation Space
        self.PropagationSpace = "Hilbert"
        if PrintDefault:
            print("\nDefine propagation space <<Hilbert>> or <<Liouville>>: PropagationSpace = ",self.PropagationSpace)

        # Acquisition Parameters
        if PrintDefault:    
            print("\nAcquisition parameters")
            print("---------------------")
        self.AcqDT = 0.0001
        if PrintDefault:
            print("Define acquisition parameter, dwell time: AcqDT = ",self.AcqDT)
        self.AcqFS = 1.0/self.AcqDT
        if PrintDefault:
            print("Define acquisition parameter, sampleing frequency: AcqFS = ",self.AcqFS)        
        self.AcqAQ = 5.0
        if PrintDefault:
            print("Define acquisition parameter, acquisition time: AcqAQ = ",self.AcqAQ)

        # Relaxation Process
        if PrintDefault:        
            print("\nRelaxation process")
            print("-----------------")
            print("\nRprocess options (Hilbert-Redfield): No Relaxation, Phenomenological, Phenomenological Matrix, "
              "\nAuto-correlated Random Field Fluctuation, Phenomenological Random Field Fluctuation"
              "\nAuto-correlated Dipolar Homonuclear, Auto-correlated Dipolar Homonuclear Ernst, Auto-correlated Dipolar Heteronuclear Ernst")
            print("\nRprocess options (liouville-Redfield): No Relaxation, Phenomenological, "
              "\nAuto-correlated Random Field Fluctuation"
              "\nAuto-correlated Dipolar Homonuclear, Auto-correlated Dipolar Homonuclear Ernst, Auto-correlated Dipolar Heteronuclear Ernst")     
            print("\nRprocess options (liouville-Lindblad): Auto-correlated Dipolar Homonuclear, Auto-correlated Dipolar Homonuclear Ernst, Auto-correlated Dipolar Heteronuclear Ernst\n") 
        self.MasterEquation = "Redfield"
        if PrintDefault:
            print("Master equation <<Redfield>> or <<Lindblad>>: MasterEquation = ",self.MasterEquation)        
        self.Rprocess = "No Relaxation" 
        if PrintDefault:                  
            print("Define relaxation process: Rprocess = ",self.Rprocess)
        self.R1 = 0.0
        if PrintDefault:
            print("Define longitudinal relaxation rate (phenominological): R1 = ",self.R1)
        self.R2 = 0.0
        if PrintDefault:
            print("Define transversel relaxation rate (phenominological): R2 = ",self.R2)
        self.R_Matrix = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        if PrintDefault:
            print("\nDefine relaxation rate matrix (phenominological): R_Matrix = \n",self.R_Matrix)
        self.RelaxParDipole_tau = 0.0
        if PrintDefault:
            print("\nDipolar relaxation parameters, Correlation time: RelaxParDipole_tau = ",self.RelaxParDipole_tau)
        self.RelaxParDipole_bIS = []
        if PrintDefault:
            print("Dipolar relaxation parameters, dipole coupling constant: RelaxParDipole_bIS = ",self.RelaxParDipole_bIS)
        self.Lindblad_Temp = 300
        if PrintDefault:
            print("\nLindblad master equation, temperature: Lindblad_T = ", self.Lindblad_Temp)

        # Maser Temperature Gradient / Repolarization
        self.InverseSpinTemp = False
        if PrintDefault:
            print("\nLindblad master equation, Inverse spin temperature: InverseSpinTemp = ", self.InverseSpinTemp)         
        self.Maser_TempGradient = False   
        if PrintDefault:
            print("\nLindblad master equation, Maser TempGradient: Maser_TempGradient = ", self.Maser_TempGradient) 
        self.Maser_RepolarizationGradient = False   
        if PrintDefault:
            print("\nLindblad master equation, Maser Repolarization: Maser_RepolarizationGradient = ", self.Maser_RepolarizationGradient)                        
        self.Lindblad_TempGradient = 0.0    
        if PrintDefault:
            print("\nLindblad master equation, temperature Gradient (dT/dt): Lindblad_TempGradient = ", self.Lindblad_TempGradient)
        self.Lindblad_InitialTemp = 0.0    
        if PrintDefault:
            print("\nLindblad master equation, initial temperature: Lindblad_InitialTemperature = ", self.Lindblad_InitialTemp) 
        self.RePol_Matrix = np.zeros((self.Vdim,self.Vdim),dtype=np.double)           
        if PrintDefault:
            print("\nLindblad master equation, repolarization: RePol_Matrix = ", self.RePol_Matrix) 
        self.RePol_Rate = 0.0           
        if PrintDefault:
            print("\nLindblad master equation, repolarization rate: RePol_Rate = ", self.RePol_Rate) 
        self.Lindblad_InitialRePolRate = 0.0
        if PrintDefault:
            print("\nLindblad master equation, initial repolarization rate: Lindblad_InitialRePolRate = ", self.Lindblad_InitialRePolRate)         
        self.Lindblad_RePolRateGradient = 0.0
        if PrintDefault:
            print("\nLindblad master equation, initial repolarization rate gradient: Lindblad_RePolRateGradient = ", self.Lindblad_RePolRateGradient)         

        # Propagation method
        if PrintDefault:    
            print("\nPropagation")
            print("----------")
            print("PropagationMethod options (Hilbert): Unitary Propagator, ODE Solver, ODE Solver ShapedPulse, ODE Solver Relaxation and Phenomenological, ODE Solver Stiff RealIntegrator")
            print("\nPropagationMethod options (Liouville): Unitary Propagator, Unitary Propagator Sparse, Relaxation, Relaxation Sparse,"
              "\nRelaxation Lindblad, Relaxation Lindblad Sparse, ODE Solver")  
        self.PropagationMethod = "ODE Solver"
        if PrintDefault:  
            print("\nDefine propagation method: PropagationMethod = ",self.PropagationMethod)

        # ODE Methods
        if PrintDefault:      
            print("\nODE solver parameters")
            print("--------------------")
        self.OdeMethod = 'RK45'
        if PrintDefault:
            print("Method used while solving the ordinary differential equation (ODE): OdeMethod = ",self.OdeMethod)
        self.ODE_atol = 1.0e-10
        if PrintDefault:
            print("ODE atol: ODE_atol = ",self.ODE_atol)
        self.ODE_rtol = 1.0e-10
        if PrintDefault:
            print("ODE rtol: ODE_rtol = ",self.ODE_rtol)

        # Radiation Damping
        if PrintDefault:    
            print("\nRadiation damping parameters")
            print("--------------------")       
        self.Rdamping = False 
        if PrintDefault:
            print("Do you want radation damping: Rdamping = ",self.Rdamping)
        self.RDxi = [0] * self.Nspins
        if PrintDefault:
            print("Radiation damping, gain: RDxi = ", self.RDxi)
        self.RDphase = [0] * self.Nspins
        if PrintDefault:
            print("Radiation damping, phase: RDRDphase = ", self.RDphase)

        # Gaussian Noise with radiation damping
        if PrintDefault:    
            print("\nGaussian noise parameters")
            print("--------------------")        
        self.NGaussian = False
        if PrintDefault:
            print("Do you want Gaussian noise with radation damping: NGaussian = ",self.NGaussian)
        self.N_mean = 0.0
        if PrintDefault:
            print("Gaussian noise, mean: N_mean = ", self.N_mean)
        self.N_std = 1.0e-8
        if PrintDefault:
            print("Gaussian noise, standard deviation: N_std = ", self.N_std)
        self.N_length = 1
        
        # Plotting
        if PrintDefault:
            print("\nPlotting options")
            print("-------------")       
        self.PlotFigureSize = (5,5)
        if PrintDefault:
            print("Figure size: PlotFigureSize = ", self.PlotFigureSize)
        self.PlotFontSize = 5
        if PrintDefault:
            print("Font size: PlotFontSize = ", self.PlotFontSize)
        self.PlotXlimt = (None,None)
        if PrintDefault:
            print("plot X limit: PlotXlimt = ", self.PlotXlimt)
        self.PlotYlimt = (None,None)
        if PrintDefault:
            print("plot Y limit: PlotYlimt = ", self.PlotYlimt)
        self.PlotArrowlength = 0.5    
        if PrintDefault:
            print("Sphere plot arrow thickness: PlotArrowlength = ", self.PlotArrowlength)
        self.PlotLinwidth = 2    
        if PrintDefault:
            print("Plot linewidht (thickness) : PlotLinwidth = ", self.PlotLinwidth)

        # Dipolar Shift Parameters
        self.Shift_para =  0.0
        self.Dipole_Shift = False   
        
        # Sparse Matrix
        if PrintDefault:
            print("\nSparse Matrix")
            print("-------------")
        self.SparseM = False 
        if PrintDefault:
            print("Do you want to use sparse matrix while exponential propagation (Liouville): SparseM = ",self.SparseM)

        # Shape Pulse
        if PrintDefault:
            print("\nShape Pulse")
            print("-----------")
        self.ShapeFunc = None
        if PrintDefault:
            print("Shape pulse function: ShapeFunc = ",self.ShapeFunc)
        self.ShapeParOmega = None
        if PrintDefault:
            print("Shape pulse amplitude: ShapeParOmega = ",self.ShapeParOmega)
        self.ShapeParPhase = None
        if PrintDefault:
            print("Shape pulse phase: ShapeParPhase = ",self.ShapeParPhase)
        self.ShapeParFreq = None
        if PrintDefault:
            print("Shape pulse frequency: ShapeParFreq = ",self.ShapeParFreq)
        
        # Numpy array type and memory (https://numpy.org/devdocs/user/basics.types.html)
        self.DTYPE_C = np.csingle # np.csingle (complex 64 bit) or np.cdouble (complex 128 bit) or np.clongdouble (complex 256 bit)
        self.DTYPE_F = np.single # np.single (float 32 bit) or np.double (float 64 bit) or np.longdouble (float 128 bit)
        self.ORDER_MEMORY = "C" # Specify the memory layout of the array. "C" for C order (row major) and "F" for Fortran order (column major) 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # PyOR Version
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def PyOR_Version(self):
        """
        Print PyOR version and versions of other packages used
        """
        
        print("PyOR Python On Resonance")
        print("\nVersion: Jeener")
        print("\nMotto: Everybody can simulate NMR")
        print("\nAuthor: Vineeth Francis Thalakottoor")
        print("\nEmail: vineethfrancis.physics@gmail.com")
        print("\nPackage Versions")
        print("--------")
        print("Numpy version: ",np.__version__)
        print("Scipy version: ",scipy.__version__)
        print("Sympy version: ",sp.__version__)
        print("Numba version: ",numba.__version__)  

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Unit Conversions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def PPMtoHz(self,ppm,ref):
        """
        Convert PPM scale to Hz

        INPUT
        -----
        ppm_ : ppm (parts per million) scale
        ref_ : Reference frequency (in Hz)

        OUTPUT
        ------
        return frequency in Hz
        """
        ref = ref / (2 * np.pi)
        return ppm * ref * 1.0e-6 
    
    def HztoPPM(self,freq_sample,ref):
        """
        Convert Hz to PPM Scale

        INPUT
        -----
        freq_sample : Sample frequency - chemical shift (in Hz)
        ref : Reference frequency (in Hz)

        OUTPUT
        ------
        return frequency in ppm unit
        """   
        ref = ref / (2 * np.pi)
        return (freq_sample) / ref * 1.0e6

    def Convert_EnergyTOFreqUnits(self,H):
        """
        Convert Hamiltonian from Energy Unit to Frequency Unit

        INPUT
        -----
        H: Hamiltonian (Joules units)
        
        OUTPUT
        ------ 
        H: Hamiltonian (Angular Frequency Units)       
        """
        
        hbar = PhysicalConstants.constants("hbar")

        return H/hbar    

    def Convert_FreqUnitsTOEnergy(self,H):
        """
        Convert Hamiltonian from Frequency Unit to Energy Unit
        
        INPUT
        -----
        H: Halitonian (Angular Frequency Units)
        
        OUTPUT
        ------
        H: Hamiltonian (Joules)
        
        """
        
        hbar = PhysicalConstants.constants("hbar")

        return H * hbar   

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialize the spin operators and particle parameters
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def Initialize(self):
        """
        Initialize:
            Spin Operators
        """
        self.SpinOperator(PrintDefault=False)
        self.ParticleParameters()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Particle Parameters
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def ParticleParameters(self):    
        """
        
        """

        for sdic, sname in zip(self.SpinDic,self.SpinName):
            setattr(self, f"{sdic}", Particle.particle(sname))  

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Spin Operators
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def SpinOperatorsSingleSpin(self,X):
        """
		Generate spin operators for a given spin: Sx, Sy and Sz
		INPUT
		-----
		X : Spin quantum number
		
		OUTPUT
		------
		SingleSpin : [Sx,Sy,Sz]
		"""   

        hbar = PhysicalConstants.constants("hbar")
		  
        # 1D Array: magnetic qunatum number for spin S (order: S, S-1, ... , -S)
        ms = np.arange(X,-X-1,-1)  
        
        # Initialize Sx, Sy and Sz operators for a spin, S
        SingleSpin = np.zeros((3,ms.shape[-1],ms.shape[-1]),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        
        # Intitialze S+ and S- operators for spin, S
        Sp = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        Sn = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        
        # Calculating the <j,m'|S+|j,m> = hbar * sqrt(j(j+1)-m(m+1)) DiracDelta(m',m+1) and
        # <j,m'|S-|j,m> = hbar * sqrt(j(j+1)-m(m-1)) DiracDelta(m',m-1)  
        Id = np.identity((ms.shape[-1])) 
        
        ## Calculate DiracDelta(m',m+1)
        ## Shifter right Identity operator
        Idp = np.roll(Id,1,axis=1) 
        ## Upper triangular martix
        Idp = np.triu(Idp,k=1) 
        
        ## Calculate DiracDelta(m',m-1)
        ## Shifter left Identity operator
        Idn = np.roll(Id,-1,axis=1) 
        ## Lower triangular matrix
        Idn = np.tril(Idn,k=1) 
        
        ## Calculating S+ and S- operators for spin, S # possibility of paralellization
        for i in range(ms.shape[-1]):
            for j in range(ms.shape[-1]):
            
                # Sz operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                SingleSpin[2][i][j] = hbar * ms[j]*Id[i][j] 
                
                # S+ operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S>  
                Sp[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]+1)) * Idp[i][j] 
                # S- operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                Sn[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]-1)) * Idn[i][j] 
        
        # Sx operator
        SingleSpin[0] = hbar * (1/2.0) * (Sp + Sn) 
        # Sy operator
        SingleSpin[1] = hbar * (-1j/2.0) * (Sp - Sn)       
            
        if self.hbarEQ1:
            SingleSpin = SingleSpin / hbar
        return SingleSpin

    def SpinOperator(self,PrintDefault=False):
        """
		Generate spin operators for all spins: Sx, Sy and Sz
		INPUT
		-----
		nill
		
		OUTPUT
		------
		Sx : array [Sx of spin 1, Sx of spin 2, Sx of spin 3, ...]
		Sy : array [Sy of spin 1, Sy of spin 2, Sy of spin 3, ...]
		Sz : array [Sz of spin 1, Sz of spin 2, Sz of spin 3, ...]
		"""
		
        Sx = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY) 
        Sy = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        Sz = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        Sp = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)        
        Sm = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)   

        # Calculating Sx, Sy and Sz operators one by one
        for i in range(self.Nspins): 
            VSlist_x = [] 
            VSlist_y = []
            VSlist_z = []
            # Computing the Kronecker product of all sub Hilbert space
            for j in range(self.Nspins):  
                # Making array of identity matrix for corresponding sub vector space
                VSlist_x.append(np.identity(self.Sdim[j])) 
                VSlist_y.append(np.identity(self.Sdim[j]))
                VSlist_z.append(np.identity(self.Sdim[j]))
            
            # Replace ith identity matrix with ith Sx,Sy and Sz operators    
            VSlist_x[i] = self.SpinOperatorsSingleSpin(self.slist[i])[0]  
            VSlist_y[i] = self.SpinOperatorsSingleSpin(self.slist[i])[1]
            VSlist_z[i] = self.SpinOperatorsSingleSpin(self.slist[i])[2]
            
            # Kronecker Product Calculating
            Sx_temp_x = VSlist_x[0]
            Sy_temp_y = VSlist_y[0]
            Sz_temp_z = VSlist_z[0]
            for k in range(1,self.Nspins):
                Sx_temp_x = np.kron(Sx_temp_x,VSlist_x[k])
                Sy_temp_y = np.kron(Sy_temp_y,VSlist_y[k]) 
                Sz_temp_z = np.kron(Sz_temp_z,VSlist_z[k]) 
            Sx[i] = Sx_temp_x
            Sy[i] = Sy_temp_y
            Sz[i] = Sz_temp_z
            
            # Calculate Sp and Sm
            Sp[i] = Sx[i] + 1j * Sy[i]  # Raising operator (S+)
            Sm[i] = Sx[i] - 1j * Sy[i]  # Lowering operator (S-)
        
        self.Sx_ = Sx 
        self.Sy_ = Sy
        self.Sz_ = Sz
        self.Sp_ = Sp
        self.Sm_ = Sm

        Jsq = np.matmul(np.sum(Sx,axis=0),np.sum(Sx,axis=0)) + np.matmul(np.sum(Sy,axis=0),np.sum(Sy,axis=0)) + np.matmul(np.sum(Sz,axis=0),np.sum(Sz,axis=0))
        self.Jsq_ = Jsq
        setattr(self, f"Jsq", QunObj(Jsq,PrintDefault=PrintDefault)) # Jsquare

        # Dynamically assign the operators to the instance attributes as QunObj
        for idx, spin in enumerate(self.SpinDic):
            setattr(self, f"{spin}x", QunObj(Sx[idx],PrintDefault=PrintDefault))  # Assign Sx[i] to Ix, Sx, Mx...
            setattr(self, f"{spin}y", QunObj(Sy[idx],PrintDefault=PrintDefault))  # Assign Sy[i] to Iy, Sy, My...
            setattr(self, f"{spin}z", QunObj(Sz[idx],PrintDefault=PrintDefault))  # Assign Sz[i] to Iz, Sz, Mz...
            setattr(self, f"{spin}p", QunObj(Sp[idx],PrintDefault=PrintDefault))  # Assign Sp[i] to Ip, Sp, Mp...
            setattr(self, f"{spin}m", QunObj(Sm[idx],PrintDefault=PrintDefault))  # Assign Sm[i] to Im, Sm, Mm...

        for idx, spin in enumerate(self.SpinDic):
            setattr(self, f"{spin}id", QunObj(np.eye(self.Vdim),PrintDefault=PrintDefault))  # Identity matrix

        self.SpinOperator_Sub(PrintDefault=PrintDefault)    

    def SpinOperator_Sub(self,PrintDefault=False):
        """
        Spin Operators of sub system
        """

        for idx, spin in enumerate(self.SpinDic):
            Sx, Sy, Sz = self.SpinOperatorsSingleSpin(self.slist[idx])

            Sp = Sx + 1j * Sy
            Sm = Sx - 1j * Sy

            # Assign each component as a QunObj
            setattr(self, f"{spin}x_sub", QunObj(Sx, PrintDefault=PrintDefault))
            setattr(self, f"{spin}y_sub", QunObj(Sy, PrintDefault=PrintDefault))
            setattr(self, f"{spin}z_sub", QunObj(Sz, PrintDefault=PrintDefault))
            setattr(self, f"{spin}p_sub", QunObj(Sp, PrintDefault=PrintDefault))
            setattr(self, f"{spin}m_sub", QunObj(Sm, PrintDefault=PrintDefault))
            setattr(self, f"{spin}id_sub", QunObj(np.eye(Sx.shape[0]), PrintDefault=PrintDefault))

    def SpinOperator_SpinQunatulNumber_List(self,SpinQNlist):
        """
		Generate spin operators for all spins: Sx, Sy and Sz
		INPUT
		-----
		nill
		
		OUTPUT
		------
		Sx : array [Sx of spin 1, Sx of spin 2, Sx of spin 3, ...]
		Sy : array [Sy of spin 1, Sy of spin 2, Sy of spin 3, ...]
		Sz : array [Sz of spin 1, Sz of spin 2, Sz of spin 3, ...]
		"""

        Slist = SpinQNlist
        S = np.asarray(Slist, dtype=float)
        Nspins = S.shape[-1]
        Sdim = np.zeros(S.shape[-1],dtype='int') 
        for i in range(S.shape[-1]): 
            Sdim[i] = np.arange(-S[i],S[i]+1,1).shape[-1]
        Vdim = np.prod(Sdim)    

        # Sx operator for individual Spin, Sx[i] corresponds to ith spin
        Sx = np.zeros((Nspins,Vdim,Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY) 
        # Sy operator for individual Spin, Sy[i] corresponds to ith spin
        Sy = np.zeros((Nspins,Vdim,Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        # Sz operator for individual Spin, Sz[i] corresponds to ith spin
        Sz = np.zeros((Nspins,Vdim,Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        
        # Calculating Sx, Sy and Sz operators one by one
        for i in range(Nspins): 
            VSlist_x = [] 
            VSlist_y = []
            VSlist_z = []
            # Computing the Kronecker product of all sub Hilbert space
            for j in range(Nspins):  
                # Making array of identity matrix for corresponding sub vector space
                VSlist_x.append(np.identity(Sdim[j])) 
                VSlist_y.append(np.identity(Sdim[j]))
                VSlist_z.append(np.identity(Sdim[j]))
            
            # Replace ith identity matrix with ith Sx,Sy and Sz operators    
            VSlist_x[i] = self.SpinOperatorsSingleSpin(Slist[i])[0]  
            VSlist_y[i] = self.SpinOperatorsSingleSpin(Slist[i])[1]
            VSlist_z[i] = self.SpinOperatorsSingleSpin(Slist[i])[2]
            
            # Kronecker Product Calculating
            Sx_temp_x = VSlist_x[0]
            Sy_temp_y = VSlist_y[0]
            Sz_temp_z = VSlist_z[0]
            for k in range(1,Nspins):
                Sx_temp_x = np.kron(Sx_temp_x,VSlist_x[k])
                Sy_temp_y = np.kron(Sy_temp_y,VSlist_y[k]) 
                Sz_temp_z = np.kron(Sz_temp_z,VSlist_z[k]) 
            Sx[i] = Sx_temp_x
            Sy[i] = Sy_temp_y
            Sz[i] = Sz_temp_z
            
        return Sx, Sy, Sz

    def MagQnuSingle(self,X):
        """
        Magnetic Quantum number of individual spins
        
        INPUT
        -----
        X : Spin Quantum Number, Integer
        
        OUTPUT
        ------
        return Magnetic quantum numbers, X, X-1, X-2,.., -X
        """
        return np.arange(X,-X-1,-1)

    def MagQnuSystem(self):
        """
        Magnetic quantum number of each state; Sz | i > = m_i | i > or < i | Sz | i > = m_i
        
        INPUT
        -----
        Sz: Spin operator, Sz
        Formating of output: 'array' or 'list'
        
        OUTPUT
        ------
        Return magnetic quantum number of each state, as 'array' or 'list'
        """
        
        Sz = self.Sz_
        
        return (np.sum(Sz,axis=0).real).diagonal()     

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Quantum States
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    def StateZeeman(self, MagQunList):
        """
        Computes the Zeeman state splitting based on magnetic quantum numbers.
        Matches each magQun value to the corresponding eigval and retrieves eigvec.
        """
        # Store the dictionary keys as spin labels (e.g., ["I", "S", "M"])
        SpinDic = list(MagQunList.keys())

        # Store spin values as a NumPy array (e.g., [1/2, 1/2, 1/2])
        magQun = np.array(list(MagQunList.values()))
        #print(SpinDic,magQun)

        eigenvectors = []

        # Loop through each spin label and match eigenvalues
        for i in range(len(SpinDic)):  # Fixed: should be range(len(SpinDic))
            spin_label = SpinDic[i]  # Get current spin label (e.g., "I", "S", etc.)
            attribute_name = f"{spin_label}z_sub"  # Construct attribute name

            # Get eigenvalues and eigenvectors
            eigval, eigvec = self.Eigen_Split(getattr(self, attribute_name))

            # Find the index of magQun[i] in eigval
            index = np.where(np.isclose(eigval, magQun[i], atol=1e-8))[0]

            #print(f"Matching {magQun[i]} -> index = {index}")

            if index.size > 0:  # Ensure we found a match
                eigenvectors.append(eigvec[index[0]])  # Store the corresponding eigenvector
            #print(eigenvectors[i].matrix)    

        #print(len(eigenvectors))
        if len(eigenvectors) == 1:
            return eigenvectors[0]
        else:
            EV = eigenvectors[0]
            for i in range(len(eigenvectors)-1):
                EV = EV.Tensor(eigenvectors[i+1])
            return EV 

    def States(self, DicList):  
        """
        Processes a list of dictionaries, distinguishing between simple and nested ones.
        Under Testing
        """

        eigenvectors = []
        eigenvectors_multi = []
        
        for d in DicList:  # Iterate over each dictionary in the list
            if any(isinstance(value, dict) for value in d.values()):
                # Handle the case where the dictionary contains nested dictionaries
                #print("Nested dictionary detected:", d)
                new_dict = d["New"]
                old_dict = d["Old"]
                l_value = d["New"]["l"]
                m_value = d["New"]["m"]
                select_value = d["New"]["Select_l"]
                SpinDic = list(old_dict.keys())
                #print(SpinDic)
                New_SpinList = []
                for i in old_dict:
                    New_SpinList.append(SpinQuantumNumber.spin(self.SpinList[i]))   
                
                Sx,Sy,Sz = self.SpinOperator_SpinQunatulNumber_List(New_SpinList)

                S_ = np.matmul(np.sum(Sx,axis=0),np.sum(Sx,axis=0)) + np.matmul(np.sum(Sy,axis=0),np.sum(Sy,axis=0)) + np.matmul(np.sum(Sz,axis=0),np.sum(Sz,axis=0))
                #print(New_SpinList)

                QunObj_S = QunObj(S_)
                QunObj_Sz = QunObj(np.sum(Sz,axis=0))
                eigenvalue_, eigenvector_objs = self.Eigen_Split(QunObj_S + QunObj_Sz)
                
                #print(eigenvalue_)
                for i in eigenvector_objs:
                    ll,mm = self.State_SpinQuantumNumber_SpinOperators(i, QunObj_S, QunObj_Sz)
                    if l_value == ll and m_value == mm:
                        eigenvectors_multi.append(i) 
                eigenvectors.append(eigenvectors_multi[select_value])      
                eigenvectors_multi = []                             
            else:
                # Handle simple dictionary case
                #print("Simple dictionary:", d)
                X_ = self.StateZeeman(d)
                eigenvectors.append(X_)

        #print(len(eigenvectors))
        if len(eigenvectors) == 1:
            return eigenvectors[0]
        else:
            EV = eigenvectors[0]
            for i in range(len(eigenvectors)-1):
                EV = EV.Tensor(eigenvectors[i+1])
            return EV


    def Bracket(self, X: 'QunObj', A: 'QunObj', Y: 'QunObj') -> 'QunObj':
        """
        Computes the double commutation superoperator: [[X, rho], Y] = (X rho - rho X) Y - Y (X rho - rho X)
        
        Parameters:
        -----------
        X : QunObj
            First input quantum object representing an operator.
        Y : QunObj
            Second input quantum object representing an operator.

        Returns:
        --------
        QunObj
            Double commutation superoperator as a QunObj instance.
        """     
        if not isinstance(X, QunObj) or not isinstance(A, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")
        
        ans = np.matmul(X.data.conj().T,np.matmul(A.data,Y.data)) / np.matmul(X.data.conj().T,Y.data)
        return ans[0].real


    def State_SpinQuantumNumber_SpinOperators(self, A: 'QunObj', Ssq: 'QunObj', Sz: 'QunObj'):
        """
        Find spin quantum number from the state.
        Returns only the real and positive root of the quadratic equation.
        The returned value is rounded to the nearest integer or half-integer.
        """
        if not isinstance(A, QunObj) or not isinstance(Ssq, QunObj) or not isinstance(Sz, QunObj):
            raise TypeError("state must be instances of QunObj.")

        # Assuming self.Bracket is implemented to calculate the commutator result
        XX = self.Bracket(A, Ssq, A)  # This should return a scalar value
        magQunNumber = self.Bracket(A, Sz, A)  # Should return a scalar value as well
        magQunNumber = round(magQunNumber[0] * 2) / 2 # Round to nearest half-integer

        #print("XX",XX)

        # Coefficients for quadratic equation
        a = 1
        b = 1
        #c = -XX  # The negative of the commutator result
        c = -1 * np.round(XX, decimals=10)  # The negative of the commutator result

        # Discriminant (should be a non-negative number for real roots)
        discriminant = b**2 - 4*a*c
        #print("disc",discriminant)
        # Check if discriminant is non-negative (real roots exist)
        if discriminant >= 0:
            # Calculate the two roots
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)

            # Filter to return only real and positive roots
            real_positive_roots = [root for root in [root1, root2] if root >= 0]

            if real_positive_roots:
                # Round to the nearest integer or half-integer
                spin_quantum_number = round(real_positive_roots[0] * 2) / 2  # Round to nearest half-integer

                # Print for debugging
                #print("Spin quantum number = ", spin_quantum_number, "and magnetic quantum number = ", magQunNumber)
                
                return spin_quantum_number, magQunNumber  # Return the rounded positive real root
            else:
                print("No positive real roots found.")
                return None  # No positive real root exists
        else:
            print("Discriminant is negative, no real roots.")
            return None  # No real roots

    def State_SpinQuantumNumber(self, A: 'QunObj'):
        """
        Find spin quantum number from the state.
        Returns only the real and positive root of the quadratic equation.
        The returned value is rounded to the nearest integer or half-integer.
        """
        if not isinstance(A, QunObj):
            raise TypeError("state must be instances of QunObj.")

        # Assuming self.Bracket is implemented to calculate the commutator result
        X = self.Bracket(A, self.Jsq, A)  # This should return a scalar value
        magQunNumber = self.Bracket(A, QunObj(np.sum(self.Sz_, axis=0)), A)  # Should return a scalar value as well
        magQunNumber = round(magQunNumber[0] * 2) / 2  # Round to nearest half-integer

        #print("X",X)

        # Coefficients for quadratic equation
        a = 1
        b = 1
        c = -X  # The negative of the commutator result

        # Discriminant (should be a non-negative number for real roots)
        discriminant = b**2 - 4*a*c
        #print("disc",discriminant)
        # Check if discriminant is non-negative (real roots exist)
        if discriminant >= 0:
            # Calculate the two roots
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)

            # Filter to return only real and positive roots
            real_positive_roots = [root for root in [root1, root2] if root >= 0]

            if real_positive_roots:
                # Round to the nearest integer or half-integer
                spin_quantum_number = round(real_positive_roots[0] * 2) / 2  # Round to nearest half-integer

                # Print for debugging
                print("Spin quantum number = ", spin_quantum_number, "and magnetic quantum number = ", magQunNumber)
                
                return spin_quantum_number, magQunNumber  # Return the rounded positive real root
            else:
                print("No positive real roots found.")
                return None  # No positive real root exists
        else:
            print("Discriminant is negative, no real roots.")
            return None  # No real roots

    def Eigen(self, A: QunObj) -> QunObj:   
        """
        Eigen values and eigen vector
        """        

        if not isinstance(A, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")

        eigenvalues, eigenvectors = la.eig(A.data)  

        return eigenvalues, QunObj(eigenvectors)

    def Eigen_Split(self, A: QunObj) -> QunObj:   
        """
        Eigenvalues and eigenvectors, where eigenvectors are split into individual QunObj instances.
        """        

        if not isinstance(A, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(A.data)  

        # Split eigenvectors as individual QunObj instances, each a column vector
        eigenvector_objs = [QunObj(vec.reshape(-1, 1)) for vec in eigenvectors.T]

        return eigenvalues, eigenvector_objs
    
    def ZeemanBasis_Ket(self):
        """
        Return a list of all the Basis kets
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.slist:
            for k in self.MagQnuSingle(j):
                locals()["Spin_List_" + str(dummy)].append("|"+str(Fraction(j))+","+str(Fraction(k))+">")
            dummy = dummy + 1    
            
        def Combine(A,B):
            for l in A:
                for m in B:
                    LABEL_temp.append(l+m)
            return LABEL_temp
            
        if self.Nspins == 1:
            LABEL =  locals()["Spin_List_" + str(0)]
            return LABEL
        
        if self.Nspins >= 2:                 
            for n in range(self.Nspins - 1):
                locals()["Spin_List_" + str(n+1)] = Combine(locals()["Spin_List_" + str(n)] ,locals()["Spin_List_" + str(n+1)])   
                LABEL_temp = []        
            LABEL = locals()["Spin_List_" + str(self.Nspins - 1)]
            return LABEL   

    def ZeemanBasis_Bra(self):
        """
        Return a list of all the Basis Bras
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.slist:
            for k in self.MagQnuSingle(j):
                locals()["Spin_List_" + str(dummy)].append("<"+str(Fraction(j))+","+str(Fraction(k))+"|")
            dummy = dummy + 1    
            
        def Combine(A,B):
            for l in A:
                for m in B:
                    LABEL_temp.append(l+m)
            return LABEL_temp
            
        if self.Nspins == 1:
            LABEL =  locals()["Spin_List_" + str(0)]
            return LABEL
        
        if self.Nspins >= 2:                 
            for n in range(self.Nspins - 1):
                locals()["Spin_List_" + str(n+1)] = Combine(locals()["Spin_List_" + str(n)] ,locals()["Spin_List_" + str(n+1)])   
                LABEL_temp = []        
            LABEL = locals()["Spin_List_" + str(self.Nspins - 1)]
            return LABEL