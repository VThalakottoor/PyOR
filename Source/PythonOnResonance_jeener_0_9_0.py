"""
Python On Resonance (PyOR)
Author: Vineeth Francis Thalakottoor Jose Chacko
Email: vineethfrancis.physics@gmail.com

Version: Jeener-0.9.0

Motto: "Everybody can simulate NMR"

Origin of PyOR
--------------
I developed PyOR during the early stage of my postdoc at École Normale Supérieure, Paris, purely for pleasure, to simulate NMR masers/rasers and assist individuals from physics, chemistry, and biology backgrounds in learning and simulating NMR experiments. PyOR is written in Python because it is free and widely accessible.

PyOR is particularly useful for beginners with a basic understanding of matrices, spin operators, and Python programming who are interested in coding magnetic resonance pulse sequences and relaxation mechanics. It can also serve as an educational tool for teaching NMR to undergraduate and graduate students.

Finally, watch out for numerical inaccuracies and potential errors in equations!

Requirments
-----------
1. Anaconda (Python Distribution)
	https://www.anaconda.com/download
	
2. Visual Studio Code
	https://code.visualstudio.com/
	
3. Ipympl (Enables using the interactive features of matplotlib)
	pip install ipympl	

Prerequisite
------------
1. Basic knowledge of Python programming

2. Basic knowledge of Spin Operators
	Read Protein NMR Spectroscopy: Principles and Practice, John Cavanagh et. al.

"""

# ---------- Package

import numpy as np
from numpy import linalg as lina

import sympy as sp
from sympy import *
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

import os
import sys
#sys.setrecursionlimit(1500)

import numba
from numba import njit, cfunc

from IPython.display import display, Latex, Math

from fractions import Fraction

import re
from io import StringIO
# ---------- Package

class Numerical_MR:

    def __init__(self,Slist):
    
        # Physical Constants
        self.pl = 6.626e-34 # Planck Constant; J s
        self.hbar = 1.054e-34 # Planck Constant; J s rad^-1
        self.ep0 = 8.854e-12 # Permitivity of free space; F m^-1
        self.mu0 = 4 * np.pi * 1.0e-7 # Permeabiltiy of free space; N A^-2 or H m^-1
        self.kb = 1.380e-23 # Boltzmann Constant; J K^-1
    
        # 1D list of spin values
        self.Slist = Slist
        self.S = np.asarray(Slist) # Change self.S to self.Slist_
    
        # Number of Spins
        self.Nspins = self.S.shape[-1]
        
        # Array of dimensions of individual Hilbert Space    
        Sdim = np.zeros(self.S.shape[-1],dtype='int') 
        for i in range(self.S.shape[-1]): 
            Sdim[i] = np.arange(-self.S[i],self.S[i]+1,1).shape[-1]
        self.Sdim = Sdim    
    
        # Dimension of Hilbert Space
        self.Vdim = np.prod(self.Sdim) 
        
        # Dimenion of Liouville Space
        self.Ldim = (self.Vdim)**2
        
        # Gyromagnetic ratio
        self.gammaE = -1.761e11 # Electron; rad s^-1 T^-1
        self.gammaH1 = 267.522e6 # Proton; rad s^-1 T^-1
        self.gammaH2 = 41.065e6 # Deuterium; rad s^-1 T^-1
        self.gammaC13 = 67.2828e6 # Carbon; rad s^-1 T^-1
        self.gammaN14 = 19.311e6 # Nitrogen 14; rad s^-1 T^-1
        self.gammaN15 = -27.116e6 # Nitrogen 15; rad s^-1 T^-1
        self.gammaO17 = -36.264e6 # Oxygen 17; rad s^-1 T^-1
        self.gammaF19 = 251.815e6 # Flurine 19; rad s^-1 T^-1  
        
        # Proton Larmor Frequency
        
        self.L285 = 6.7 # T
        self.L300 = 7.04925 # T
        self.L400 = 9.39798 # T
        self.L500 = 11.7467 # T
        self.L600 = 14.0954 # T
        self.L700 = 16.4442 # T
        self.L750 = 17.6185 # T
        self.L800 = 18.7929 # T
        self.L850 = 19.9673 # T
        self.L900 = 21.1416 # T
        self.L950 = 22.3160 # T
        self.L1000 = 23.4904 # T
        
        # Nuclear Spin
        self.H1 = 1/2
        self.H2 = 1
        self.H3 = 1/2
        self.He3 = 1/2
        self.Li6 = 1
        self.Li7 = 3/2
        self.Be9 = 3/2
        self.B10 = 3
        self.B11 = 3/2
        self.C13 = 1/2
        self.N14 = 1
        self.N15 = 1/2
        self.O17 = 5/2
        self.F19 = 1/2

        print("\nPyOR default parameters/settings")
        print("--------------------------------")
                
        # NMR Simulation Parameters
        
        # Planck Constant
        self.hbarEQ1 = True
        print("\nDefine energy units: hbarEQ1 = ",self.hbarEQ1)

        # Matrix element tolarence
        self.MatrixTolarence = 1.0e-6
        print("\nDefine the matrix tolerence (make matrix elements less than tolarence value to zero): MatrixTolarence = ",self.MatrixTolarence)
        
        # Gyromagnetic Ration        
        self.Gamma = [0] * self.Nspins
        print("\nDefine the gyromagnetic ratios: Gamma = ",self.Gamma)
        
        # Spectrometer Field in T
        self.B0 = None
        print("\nDefine the static field along Z: B0 = ",self.B0)
        
        # Rotating Frame Frequency
        self.OmegaRF = [0] * self.Nspins 
        print("\nDefine rotating frame frequency: OmegaRF = ",self.OmegaRF)
        
        # Offset Frequency in rotating frame (Hz) 
        self.Offset = [0] * self.Nspins  
        print("\nDefine the offset frequencies of the spins: Offset = ",self.Offset) 

        # Larmor Frequency
        self.print_Larmor = True
        print("\nDo you want to print the larmor frequency: print_Larmor = ",self.print_Larmor)        
        
        # J Coupling 
        self.Jlist = np.zeros((self.Nspins,self.Nspins))
        print("\nDefine the J coupling: Jlist = \n",self.Jlist)
        
        # Dipole dipole pairs tuple of list, spin pair interact by dipolar coupling, useful for relaxation
        self.DipolePairs = []
        print("\nDefine the spin paris dipolar coupled: DipolePairs = ",self.DipolePairs)

        # Initial and final spin temperature
        print("\nSpin temperatures")
        print("----------------")
        self.Ispintemp = [0] * self.Nspins
        print("Initial spin temperature of individual spins: Ispintemp = ",self.Ispintemp)
        self.Fspintemp = [0] * self.Nspins
        print("Final spin temperature of individual spins = ",self.Fspintemp)

        # Propagation Space
        self.PropagationSpace = "Hilbert"
        print("\nDefine propagation space <<Hilbert>> or <<Liouville>>: PropagationSpace = ",self.PropagationSpace)

        # Acquisition Parameters
        print("\nAcquisition parameters")
        print("---------------------")
        self.AcqDT = 0.0001
        print("Define acquisition parameter, dwell time: AcqDT = ",self.AcqDT)
        self.AcqFS = 1.0/self.AcqDT
        print("Define acquisition parameter, sampleing frequency: AcqFS = ",self.AcqFS)        
        self.AcqAQ = 5.0
        print("Define acquisition parameter, acquisition time: AcqAQ = ",self.AcqAQ)

        # Relaxation Process
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
        print("Master equation <<Redfield>> or <<Lindblad>>: MasterEquation = ",self.MasterEquation)        
        self.Rprocess = "No Relaxation"                   
        print("Define relaxation process: Rprocess = ",self.Rprocess)
        self.R1 = 0.0
        print("Define longitudinal relaxation rate (phenominological): R1 = ",self.R1)
        self.R2 = 0.0
        print("Define transversel relaxation rate (phenominological): R2 = ",self.R2)
        self.R_Matrix = np.zeros((self.Vdim,self.Vdim),dtype=np.double)

        print("\nDefine relaxation rate matrix (phenominological): R_Matrix = \n",self.R_Matrix)
        self.RelaxParDipole_tau = 0.0
        print("\nDipolar relaxation parameters, Correlation time: RelaxParDipole_tau = ",self.RelaxParDipole_tau)
        self.RelaxParDipole_bIS = []
        print("Dipolar relaxation parameters, dipole coupling constant: RelaxParDipole_bIS = ",self.RelaxParDipole_bIS)
        self.Lindblad_T = 300
        print("\nLindblad master equation, temperature: Lindblad_T = ", self.Lindblad_T)


        # Propagation method
        print("\nPropagation")
        print("----------")
        print("PropagationMethod options (Hilbert): Unitary Propagator, ODE Solver, ODE Solver ShapedPulse, ODE Solver Relaxation and Phenomenological, ODE Solver Stiff RealIntegrator")
        print("\nPropagationMethod options (Liouville): Unitary Propagator, Unitary Propagator Sparse, Relaxation, Relaxation Sparse,"
              "\nRelaxation Lindblad, Relaxation Lindblad Sparse, ODE Solver")
        self.PropagationMethod = "ODE Solver"
        print("Define propagation method: PropagationMethod = ",self.PropagationMethod)

        # ODE Methods
        print("\nODE solver parameters")
        print("--------------------")
        self.OdeMethod = 'RK45'
        print("Method used while solving the ordinary differential equation (ODE): OdeMethod = ",self.OdeMethod)
        self.ODE_atol = 1.0e-13
        print("ODE atol: ODE_atol = ",self.ODE_atol)
        self.ODE_rtol = 1.0e-13
        print("ODE rtol: ODE_rtol = ",self.ODE_rtol)
        
        # Plotting
        print("\nPlotting options")
        print("-------------")       
        self.PlotFigureSize = (5,5)
        print("Figure size: PlotFigureSize = ", self.PlotFigureSize)
        self.PlotFontSize = 5
        print("Font size: PlotFontSize = ", self.PlotFontSize)
        self.PlotXlimt = (None,None)
        print("plot X limit: PlotXlimt = ", self.PlotXlimt)
        self.PlotYlimt = (None,None)
        print("plot Y limit: PlotYlimt = ", self.PlotYlimt)
        
        # Dipolar Shift Parameters
        self.Shift_para =  0.0
        self.Dipole_Shift = False   
        
        # Sparse Matrix
        print("\nSparse Matrix")
        print("-------------")
        self.SparseM = False 
        print("Do you want to use sparse matrix while exponential propagation (Liouville): SparseM = ",self.SparseM)

        # Shape Pulse
        print("\nShape Pulse")
        print("-----------")
        self.ShapeFunc = None
        print("Shape pulse function: ShapeFunc = ",self.ShapeFunc)
        self.ShapeParOmega = None
        print("Shape pulse amplitude: ShapeParOmega = ",self.ShapeParOmega)
        self.ShapeParPhase = None
        print("Shape pulse phase: ShapeParPhase = ",self.ShapeParPhase)
        self.ShapeParFreq = None
        print("Shape pulse frequency: ShapeParFreq = ",self.ShapeParFreq)
        
        # Numpy array type and memory (https://numpy.org/devdocs/user/basics.types.html)
        self.DTYPE_C = np.csingle # np.csingle (complex 64 bit) or np.cdouble (complex 128 bit) or np.clongdouble (complex 256 bit)
        self.DTYPE_F = np.single # np.single (float 32 bit) or np.double (float 64 bit) or np.longdouble (float 128 bit)
        self.ORDER_MEMORY = "C" # Specify the memory layout of the array. "C" for C order (row major) and "F" for Fortran order (column major)  

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # PyOR Version
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def PyOR_About(self):
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
        
        return H/self.hbar    

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
        
        return H * self.hbar
           
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Functions to generate the Spin System
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
                SingleSpin[2][i][j] = self.hbar * ms[j]*Id[i][j] 
                
                # S+ operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S>  
                Sp[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]+1)) * Idp[i][j] 
                # S- operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                Sn[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]-1)) * Idn[i][j] 
        
        # Sx operator
        SingleSpin[0] = self.hbar * (1/2.0) * (Sp + Sn) 
        # Sy operator
        SingleSpin[1] = self.hbar * (-1j/2.0) * (Sp - Sn)       
            
        if self.hbarEQ1:
            SingleSpin = SingleSpin / self.hbar
        return SingleSpin
        
    def SpinOperator(self):
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
		
        # Sx operator for individual Spin, Sx[i] corresponds to ith spin
        Sx = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY) 
        # Sy operator for individual Spin, Sy[i] corresponds to ith spin
        Sy = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        # Sz operator for individual Spin, Sz[i] corresponds to ith spin
        Sz = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        
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
            VSlist_x[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[0]  
            VSlist_y[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[1]
            VSlist_z[i] = self.SpinOperatorsSingleSpin(self.Slist[i])[2]
            
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
            
        return Sx, Sy, Sz
        
    def PMoperators(self,Sx,Sy):
        """
		Generate spin operators for all spins: Sp (Sx + j Sy) and Sm (Sx - j Sy)
		INPUT
		-----
		Sx, Sy
		
		OUTPUT
		------
		Sp : array [Sp of spin 1, Sp of spin 2, Sp of spin 3, ...]
		Sm : array [Sm of spin 1, Sm of spin 2, Sm of spin 3, ...]
		"""
		    
        Sp = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        Sm = np.zeros((self.Nspins,self.Vdim,self.Vdim),dtype=self.DTYPE_C,order=self.ORDER_MEMORY)
        for i in range(self.Nspins):
            Sp[i] = Sx[i] + 1j * Sy[i]
            Sm[i] = Sx[i] - 1j * Sy[i]
  
        return Sp, Sm   

    def GenerateSpinOperators(self):
        """
        Generate spin operators for all spins: Sx, Sy and Sz
        INPUT
        -----
        Nspins  : Number of Spins
        Vdim    : Dimension of Hilbert Space
        Sdim    : Array of dimensions of individual Hilbert Space
        Slist   : 1D list of spin values
        hbarEQ1 : True (unit of hamiltonian is angular frequency). False (unit of Hamiltonian is Joule).

        OUTPUT
        ------
        Sx : array [Sx of spin 1, Sx of spin 2, Sx of spin 3, ...]
        Sy : array [Sy of spin 1, Sy of spin 2, Sy of spin 3, ...]
        Sz : array [Sz of spin 1, Sz of spin 2, Sz of spin 3, ...]
        Sp : array [Sp of spin 1, Sp of spin 2, Sp of spin 3, ...]
        Sm : array [Sm of spin 1, Sm of spin 2, Sm of spin 3, ...]
        """  
        Sx_, Sy_, Sz_ = self.SpinOperator() 
        self.Sx_ = Sx_
        self.Sy_ = Sy_
        self.Sz_ = Sz_
        
        Sp_, Sm_ = self.PMoperators(Sx_,Sy_)
        self.Sp_ = Sp_
        self.Sm_ = Sm_ 
        
        return Sx_, Sy_, Sz_, Sp_, Sm_

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Basis State of the system
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    def MagQnu(self,X):
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

    def Basis_Ket_AngularMomentum_Array(self):
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

    def Basis_Ket_AngularMomentum_List(self):
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
            
        array = (np.sum(Sz,axis=0).real).diagonal()
        List = []
        for i in array:
            List.append(str(Fraction(float(i))))
        return List
        
    def Basis_Ket(self):
        """
        Return a list of all the Basis kets
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.Slist:
            for k in self.MagQnu(j):
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

    def Basis_Bra(self):
        """
        Return a list of all the Basis Bras
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()["Spin_List_"+str(i)] = []
        dummy = 0
        for j in self.Slist:
            for k in self.MagQnu(j):
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

    def ZBasis_H(self,Hz):
        """"
        Zeeman Basis
        INPUT
        -----
        Hz: Zeman Hamiltonian (lab frame)
        
        OUTPUT
        ------
        return BZ (eigen vectors of Zeman Hamiltonian (lab frame): Bz[0] first eigen vector, Bz[1] second eigen vector, ... )
        """
        
        B_Zeeman = []
        eigenvalues, eigenvectors = lina.eig(Hz) 
        for i in range(self.Vdim):
            B_Zeeman.append((eigenvectors[:,i].reshape(-1,1)).real)         
        return B_Zeeman 
        
    def STBasis(self,Hz):
        """
        Singlet Triplet Basis (Two Spin Half Only)
        INPUT
        -----
        Hz: Zeman Hamiltonian (lab frame)
        
        OUTPUT
        ------
        return Singlet Triplet basis      
        """
        
        if ((self.Nspins == 2) and (self.S[0] == 1/2) and (self.S[1] == 1/2)):
            B_Zeeman = self.ZBasis_H(Hz)
            B_ST = []
            B_ST.append(B_Zeeman[0]) # Tm
            B_ST.append((1/np.sqrt(2)) * (B_Zeeman[1] + B_Zeeman[2])) 
            B_ST.append(B_Zeeman[3])
            B_ST.append((1/np.sqrt(2)) * (B_Zeeman[1] - B_Zeeman[2]))  
            display(Latex(r'Basis: $T_{-}$, $T_{0}$,$T_{+}$,$S_{0}$'))  
            return B_ST
        else:
            print("Two spin half system only")
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Halitonian of the Spin System
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
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
        Gamma = self.Gamma
        B0 = self.B0
        Offset = self.Offset
        
        W0 = np.zeros((self.Nspins))
        gamma = np.asarray(Gamma)
        offset = np.asarray(Offset)
        for i in range(self.Nspins):
            W0[i] = -1 * gamma[i] * B0 - 2 * np.pi * offset[i]
        
        self.LarmorF = W0

        if self.print_Larmor:
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
        Sz = self.Sz_
        
        Hz = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            Hz = Hz + LarmorF[i] * Sz[i]
                
        return Hz

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
        OmegaRF = self.OmegaRF
        Sz = self.Sz_
        
        omegaRF = np.asarray(OmegaRF)
        Hz = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            Hz = Hz + (LarmorF[i]-omegaRF[i]) * Sz[i]
        return Hz
                
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
        
        Sx = self.Sx_
        Sy = self.Sy_ 
        
        HzB1 = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        omega1 = 2*np.pi*Omega1
        Omega1Phase = np.pi*Omega1Phase/180.0
        for i in range(self.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1Phase) + Sy[i]*np.sin(Omega1Phase))
        return HzB1
        
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

        Sx = self.Sx_
        Sy = self.Sy_ 
        
        HzB1 = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        omega1 = 2*np.pi*Omega1
        Omega1freq = 2*np.pi*Omega1freq
        Omega1Phase = np.pi*Omega1Phase/180.0
        for i in range(self.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1freq * t + Omega1Phase) + Sy[i]*np.sin(Omega1freq * t + Omega1Phase))
        return HzB1        

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

        Sx = self.Sx_
        Sy = self.Sy_ 
        
        HzB1 = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        omega1 = 2*np.pi*(Omega1T(t))
        Omega1freq = 2*np.pi*Omega1freq
        Omega1Phase = (Omega1PhaseT(t))
        for i in range(self.Nspins):
            HzB1 = HzB1 + omega1 * (Sx[i]*np.cos(Omega1freq * t + Omega1Phase) + Sy[i]*np.sin(Omega1freq * t + Omega1Phase))
        return HzB1 
        
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
        
        J = self.Jlist
        Sx = self.Sx_
        Sy = self.Sy_ 
        Sz = self.Sz_
        
        J = np.triu(2*np.pi*J)    
        Hj = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            for j in range(self.Nspins):
                Hj = Hj + J[i][j] * (np.matmul(Sx[i],Sx[j]) + np.matmul(Sy[i],Sy[j]) + np.matmul(Sz[i],Sz[j]))      
        return Hj        

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
        
        J = self.Jlist
        Sz = self.Sz_
        
        J = np.triu(2*np.pi*J)    
        Hj = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i in range(self.Nspins):
            for j in range(self.Nspins):
                Hj = Hj + J[i][j] * np.matmul(Sz[i],Sz[j])      
        return Hj 
        
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
                
    def DDcoupling(self,bIS,theta,phi,secular):
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

        Sx = self.Sx_
        Sy = self.Sy_ 
        Sz = self.Sz_
        Sp = self.Sp_
        Sm = self.Sm_
        
        theta = np.pi*theta/180.0
        phi = np.pi*phi/180.0
        
        Spin1, Spin2 = np.array(self.DipolePairs).T
        
        Hdd = np.zeros((self.Vdim,self.Vdim),dtype=np.double)
        for i,j in zip(Spin1, Spin2):
        
            if secular:
                A = np.matmul(Sz[i],Sz[j]) * (3 * (np.cos(theta))**2 - 1)
                Hdd = Hdd + 2.0*np.pi * bIS * A
            else:     
                A = np.matmul(Sz[i],Sz[j]) * (3 * (np.cos(theta))**2 - 1)
                B = (-1/4) * (np.matmul(Sp[i],Sm[j]) + np.matmul(Sm[i],Sp[j])) * (3 * (np.cos(theta))**2 - 1)
                C = (3/2) * (np.matmul(Sp[i],Sz[j]) + np.matmul(Sz[i],Sp[j])) * np.sin(theta) * np.cos(theta) * np.exp(-1j*phi)
                D = (3/2) * (np.matmul(Sm[i],Sz[j]) + np.matmul(Sz[i],Sm[j])) * np.sin(theta) * np.cos(theta) * np.exp(1j*phi)
                E = (3/4) * np.matmul(Sp[i],Sp[j]) * (np.sin(theta))**2 * np.exp(-1j * 2 * phi)
                F = (3/4) * np.matmul(Sm[i],Sm[j]) * (np.sin(theta))**2 * np.exp(1j * 2 * phi)
                Hdd = Hdd + 2.0*np.pi * bIS * (A+B+C+D+E+F)                
            
        return Hdd 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Shaped Pulse 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def ShapedPulse_Bruker(self,file_path,pulseLength,RotationAngle):
        """
        Call shaped pulse file from Bruker
        
        INPUT
        -----
        file_path: path to shapeed pulse file
        pulseLenght: Pulse length in s
        pulseLength360_hard: Pulse length for hard pulse 360 degree
        
        OUTPUT
        ------
        pulseShapeInten: Shaped pulse intensity
        pulseShapePhase: Shaped pulse phase
        
        Reference
        ---------
        1) https://github.com/modernscientist/modernscientist.github.com/blob/master/notebooks/NMRShapedPulseSimulation.ipynb
           License: BSD (C) 2013, Michelle L. Gill
           http://themodernscientist.com/posts/2013/2013-06-09-simulation_of_nmr_shaped_pulses/
        2) Bruker Shape Tool manual
        """
        
        nu1_hard_pulseLength = RotationAngle/(360.0 * pulseLength) # Nutation frequency of hard pulse for given pulse length and rotation angle
        print("Nutation frequency of hard pulse for given pulse length and rotation angle: ",nu1_hard_pulseLength)
        
        pulseString = open(file_path, 'r').read()
        pulseShapeArray = np.genfromtxt(StringIO(pulseString), comments='#', delimiter=',')
        
        n_pulse = pulseShapeArray.shape[0] # Number of pulses
        
        pulseShapeInten = pulseShapeArray[:,0] / np.max(np.abs(pulseShapeArray[:,0]))
        pulseShapePhase = pulseShapeArray[:,1] * np.pi/180

        xPulseShape = pulseShapeInten * np.cos(pulseShapePhase)
        yPulseShape = pulseShapeInten * np.sin(pulseShapePhase)
        
        scalingFactor = np.sum(xPulseShape)/n_pulse
        
        print("Scaling Factor: ",scalingFactor)
        
        nuB1max = nu1_hard_pulseLength / scalingFactor 
        
        print("Maximum nuB1: ",nuB1max)
        print("Period corresponding to maximum nuB1: ",1.0/nuB1max)
        
        time = np.linspace(0, pulseLength, n_pulse)
        
        return time, nuB1max * pulseShapeInten, pulseShapePhase
        
    def ShapedPulse_Interpolate(self,time,SPIntensity,SPPhase,Kind):
        """
        Interpolate the amplitude and phase
        """
        
        return interp1d(time, SPIntensity, kind=Kind, fill_value="extrapolate"), interp1d(time, SPPhase, kind=Kind, fill_value="extrapolate")            

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotation Matrix 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def RotateX(self,theta):
        """
        Rotation about X
        """
        theta = theta * np.pi / 180.0
        return np.asarray([[1,0,0],[0, np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

    def RotateY(self,theta):
        """
        Rotation about Y
        """
        theta = theta * np.pi / 180.0
        return np.asarray([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        
    def RotateZ(self,theta):
        """
        Rotation about Z
        """
        theta = theta * np.pi / 180.0
        return np.asarray([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

    def RotateEuler(self,alpha,beta,gamma):
        """
        Euler Angles
        """
        return self.RotateZ(alpha) @ self.RotateY(beta) @ self.RotateZ(gamma)
                
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Eigen Values and Vectors 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def Eigen(self,H):
        """
        Eigen Values and Vectors
        
        INPUT
        -----
        H: Hamiltonian
        
        OUTPUT
        ------
        return eigenvalues, eigenvectors 
        """
        
        eigenvalues, eigenvectors = lina.eig(H)
        
        return eigenvalues, eigenvectors    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Basis Transformation Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Transform_StateBasis(self,old,new):
        """
        Change Basis state from one to other: Function return the transformation matrix
        | new > = U | old >
        O_new = U O_old U_dagger
        INPUT
        -----
        old: array of old Basis state
        New: array of new Baisis state
        
        OUTPUT
        ------
        return transformation matrix
        """
        
        dim = len(old)
        U = np.zeros((dim,dim),dtype=np.cdouble)
        for i in range(dim):
            for j in range(dim):
                U[i][j] = self.Adjoint((old[i])) @ (new[j])
        return U 

    def State_BasisChange(self,state,U):
        """
        Change basis state
        
        INPUT
        -----
        state: State in old basis
        U: Transformation matrix
        
        OUTPUT
        ------
        return state in new basis
        
        """
        
        return U @ state
        
    def Operator_BasisChange(self,O,U):
        """
        Change the Operator basis
        
        INPUT
        -----
        O: Old operator
        U: Basis Transformation matric
        
        OUTPUT
        ------
        return basis transformed operator
        """
        
        return self.Adjoint(U) @ O @ U 

    def SpinOperator_BasisChange(self,Sop,U):
        """
        Change the basis of Spin Operator
        
        INPUT
        -----
        O: array of old spin operators
        U: Basis Transformation matric
        
        OUTPUT
        ------
        return transformed spin operator
        """
        
        dim = Sop.shape[0]
        Sop_N = np.zeros(Sop.shape,dtype=complex)
        for i in range(dim):
            Sop_N[i] = U @ Sop[i] @ self.Adjoint(U) 
        return Sop_N                
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Matrix Functions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Create_DensityMatrix(self,state):
        """
        Create density matrix from the state
        """
        
        return np.outer(state,Adjoint(state))

    def Norm_Matrix(self,A):
        """
        Matrix Norm (Frobenius)
        """
        
        return np.linalg.norm(A,ord='fro')

    def Adjoint(self,A):
        """
        Return adjoint of operator A

        INPUT
        -----
        A: an operator or vector
        
        OUTPUT
        ------  
        return adjoint of operator or vector      
        """
        
        return A.T.conj()

    def InnerProduct(self,A,B):
        """
        Inner Product
        
        INPUT
        -----
        A: Operator or vector
        B: Operator or vector
        
        OUTPUT
        ------
        return iiner product of the operators or vectors        
        """
        return np.trace(np.matmul(A.T.conj(),B))

    def Normalize(self,A):
        """
        Normalize 
        
        INPUT
        -----
        A: Operator
        OUTPUT
        ------        
        return normalized operator, that means: inner product of A and A equals 1
        """
        return A/np.sqrt(self.InnerProduct(A,A))
        
    def DensityMatrix_Components(self,A,rho):
        """
        Components of density matrix in the Basis Operators, inner product of A_i and rho, where A_i is the ith basis operator
        
        INPUT
        -----
        A: Basis operators (array containing basis operators)
        rho: Density matrix
        OUTPUT
        ------        
        return projection or component of density matrix in the Basis Operators         
        """
        no_Basis = A.shape[0]
        components = np.zeros((no_Basis),dtype=np.cdouble)    
        for i in range(no_Basis):
            components[i] = self.InnerProduct(A[i],rho)
        tol = 1.0e-5 # make elements lower than 'tol' into zero 
        components.real[abs(components.real) < tol] = 0.0  
        components.imag[abs(components.real) < tol] = 0.0  
        return np.round(components.real,3)

    def DensityMatrix_Components_Dictionary(self,A,dic,rho):
        """
        Components of density matrix in the Basis Operators, inner product of A_i and rho, where A_i is the ith basis operator
        
        INPUT
        -----
        A: Basis operators (array containing basis operators)
        dic: Dictionary of Spin Operators
        rho: Density matrix
        OUTPUT
        ------        
        return projection or component of density matrix in the Basis Operators         
        """

        tol = self.MatrixTolarence
        no_Basis = np.asarray(A).shape[0]
        components = np.zeros((no_Basis),dtype=np.cdouble)    
        for i in range(no_Basis):
            components[i] = self.InnerProduct(A[i],rho)
        tol = 1.0e-10 # make elements lower than 'tol' into zero 
        components.real[abs(components.real) < tol] = 0.0  
        components.imag[abs(components.real) < tol] = 0.0 
        density_out = ["Density Matrix = "]
        for i in range(no_Basis):
            if components[i].real == 0: # print only non-zero terms
                pass
            else:    
                density_out.append(str(round(components[i].real,5)) + " " + dic[i] + " + ") 
        print((''.join(density_out))[:-3])
        
    def Matrix_Tol(self,M):
        """
        Make very small values of a matrix to zero
        
        INPUT
        -----
        M: Matrix
        tol: Tolarance, below which matrix element will be zero
        
        OUTPUT
        ------
        return new matrix 
        """       
        
        tol = self.MatrixTolarence
        M.real[abs(M.real) < tol] = 0.0
        M.imag[abs(M.imag) < tol] = 0.0
        return M
        
    def Matrix_Round(self,M,roundto):
        """
        Evenly round the matrix elemnt to the given number of decimals. 
        
        INPUT
        -----
        roundto: Number of decimal places to round to
        """    
        
        return np.round(M,roundto)          

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Operator Basis Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def CG_Coefficient(self,j1,m1,j2,m2,J,M):
        """
        Clebsch-Gordan Coefficients
        
        INPUT
        -----
        j1: spin quantum number particle 1
        m1: magnetic quantum number particle 1
        j2: spin quantum number particle 2
        m2: magnetic quantum number particle 2    
        J: Total spin quantum number
        M: total magnetic quantum number    
        OUTPUT
        ------  
        return Clebsch-Gordan Coefficients        
        """
        
        return float(CG(j1, m1, j2, m2, J, M).doit())
        

    def Spherical_OpBasis(self,S):
        """
        Spherical Operator Basis
        
        INPUT
        -----
        S: spin quantum number
        
        OUTPUT
        ------        
        return spherical operator basis,Coherence order and LM_state
        """
        
        states = int(2 * S + 1)  # Number of spherical operators in Hilbert-Schidth space, states**2
        EYE = np.eye(states)
        std_basis = np.zeros((states,states,1))
        for i in range(states): 
            std_basis[i] = EYE[:,i].reshape(-1,1)
        L = np.arange(0,2*S+1,1,dtype=np.int16)
        m = -1*np.arange(-S,S+1,1,dtype=np.double)
        Pol_Basis = []
        Coherence_order = []
        LM_state = []

        for i in L:
            M = np.arange(-i,i+1,1,dtype=np.int16)
            for j in M:  
                Sum = 0
                for k in range(states):
                    for l in range(states):
                        cg_coeff = float(CG(S, m[l], i, j, S, m[k]).doit())
                        Sum = Sum + cg_coeff * np.outer(std_basis[k],std_basis[l].T.conj())
                Pol_Basis.append(np.sqrt((2*i + 1)/(2*S+1)) * Sum)
                Coherence_order.append(j) 
                LM_state.append(tuple([i,j]))
        
        print("Coherence Order: ",Coherence_order)
        print("LM state: ",LM_state)
        return Pol_Basis,Coherence_order,LM_state                 

    def ProductOperator(self,OP1,CO1,DIC1,OP2,CO2,DIC2,sort,indexing):
        """
        Product of two spherical basis operators (kronecker porduct)
        
        INPUT
        -----
        OP1 and OP2: Individual operator basis of each particles
        CO1 and CO2: Individual coherence order of each particle
        DIC1 and DIC2: Individual labels of basis operators of each particle
        sort: sort coherence order by 'normal' or 'negative to positive' or 'zero to high'
        indexing: if True, index will be added with the labelling of basis operators
        
        OUTPUT
        ------        
        OP: New Operator basis
        CO: New coherence order
        DIC: New labelling
        """
        
        CO = []
        OP = []
        DIC = []
        index = 0
        for i,j,k in zip(OP1,CO1,DIC1):
            for m,n,o in zip(OP2,CO2,DIC2):
                OP.append(np.kron(i,m))
                CO.append(j+n)
                DIC.append(k+o)
                
        if sort == 'normal':
            pass
            
        if sort == 'negative to positive':        
            # Sorting increasing coherence order
            combine = list(zip(CO,OP,DIC))
            combine_sort = sorted(combine, key=lambda x: x[0])
            Sort_CO,Sort_OP,Sort_DIC = zip(*combine_sort)  
            CO = list(Sort_CO)
            OP = list(Sort_OP)
            DIC = list(Sort_DIC)      
            
        if sort == 'zero to high':        
            # Sorting increasing coherence order
            combine = list(zip(list(map(abs, CO)),CO,OP,DIC))
            combine_sort = sorted(combine, key=lambda x: x[0])
            Sort_CO_dumy,Sort_CO,Sort_OP,Sort_DIC = zip(*combine_sort)  
            CO = list(Sort_CO)
            OP = list(Sort_OP)
            DIC = list(Sort_DIC)      
            
        if indexing:                        
            for p in range(len(DIC)):
                DIC[p] = DIC[p] + "[" + str(index) + "]"      
                index = index + 1  
                
        return OP, CO, DIC                                 

    def ProductOperators_SpinHalf_Cartesian(self,Index,Normal):
        """
        Create product operators for arbitrary spin half particles in Cartesina basis
        
        INPUT
        -----
        Nill

        OUTPUT
        ------
        return product operator basis, coherence order and dictionary
        """ 

        Dic = ["Id ","Ix ","Iy ","Iz "]
        Single_OP = self.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [np.eye(2),Single_OP[0],Single_OP[1],Single_OP[2]]
        
        Coherence_order_SpinHalf = list(range(len(Dic)))
        
        Basis_SpinHalf_out = []
        Dic_out = []
        Coherence_order_SpinHalf_out = []
                
        if self.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Dic_out = Dic
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            sort = 'normal'
            for i in range(self.Nspins-1):
                if i == self.Nspins-2:
                    indexing = Index
                if i == 0:    
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic] 
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]     
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,Basis_SpinHalf, Coherence_order_SpinHalf, Dic,sort,indexing)                
        
        if Normal:
            for j in range(self.Ldim):
                Basis_SpinHalf_out[j] = self.Normalize(Basis_SpinHalf_out[j])
        
        return Basis_SpinHalf_out, Dic_out 

    def ProductOperators_SpinHalf_PMZ(self,sort,Index,Normal):
        """
        Create product operators for arbitrary spin half particles in Cartesina basis
        
        INPUT
        -----
        Nill

        OUTPUT
        ------
        return product operator basis, coherence order and dictionary
        
        Reference: Protein NMR Spectroscopy, Principles and Practice, John Cavanagh, et.al, P 67, ed1.
        """ 

        Dic = ["Im ","Iz ","Id ","Ip "]
        Single_OP = self.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [Single_OP[0] - 1j * Single_OP[1],Single_OP[2],np.eye(2),-1 * (Single_OP[0] + 1j * Single_OP[1])]
        
        Coherence_order_SpinHalf = [-1,0,0,1]
        
        Basis_SpinHalf_out = []
        Dic_out = []
        Coherence_order_SpinHalf_out = []
                
        if self.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Dic_out = Dic
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            for i in range(self.Nspins-1):
                if i == self.Nspins-2:
                    indexing = Index
                if i == 0:    
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic] 
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]    
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,Basis_SpinHalf, Coherence_order_SpinHalf, Dic,sort,indexing)                

        if Normal:
            for j in range(self.Ldim):
                Basis_SpinHalf_out[j] = self.Normalize(Basis_SpinHalf_out[j])

        return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out 

    def ProductOperators_SpinHalf_SphericalTensor(self,sort,Index):
        """
        Create product operators for arbitrary spin half particles in Spherical Tensor operator basis
        
        INPUT
        -----
        Nill

        OUTPUT
        ------
        return product operator basis, coherence order and dictionary
        """        
        
        Dic = ["Id ","Im ","Iz ","Ip "]
        Basis_SpinHalf, Coherence_order_SpinHalf, LM_state_SpinHalf = self.Spherical_OpBasis(1/2)
        Basis_SpinHalf_out = []
        Coherence_order_SpinHalf_out = []
        Dic_out = []
        
        if self.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            
            for i in range(self.Nspins-1):    
                if i == self.Nspins-2:
                    indexing = Index
                if i == 0:    
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic] 
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]      
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,Basis_SpinHalf, Coherence_order_SpinHalf, Dic,sort,indexing)
                
        return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out 
        
    def String_to_Matrix(self, dic, Basis):
        """
        Make a relation between Basis dictionar and matrix
        
        INPUT
        -----
        dic: Dictionary of operator basis
        Basis: List of basis
        
        OUTPUT
        ------
        
        """    

        char_to_remove = "Id"
        dic = [re.sub(f"{re.escape(char_to_remove)}.", " ", s) for s in dic]               
        dic = [s.replace(" ", "") for s in dic]

        print(dic)
        return dict(zip(dic, Basis))
        
    def ProductOperators_Zeeman(self,Sz,Hz):
        """
        Projection Operators (Alpha Beta Operator Basis)
        
        INPUT
        -----
        Sz: Spin operator
        Hz: Zeeman Hamiltonian in Labframe
        OUTPUT
        ------        
        return product operator basis in Zeeman basis, dictionary, coherence order, coherence order in 2d matrix format
        """
        
        B_Z = self.ZBasis_H(Hz)
        Kets = self.Basis_Ket()
        Bras = self.Basis_Bra()
        dic = []
        coh = []
        State_Momentum = self.Basis_Ket_AngularMomentum_Array()
        
        B = []
        k = 0
        for i in range(self.Vdim):
            for j in range(self.Vdim):
                B.append(np.outer(B_Z[i],self.Adjoint(B_Z[j])))
                dic.append(Kets[i]+Bras[j])
                coh.append(State_Momentum[i] - State_Momentum[j])
                k = k + 1
   
        return B, dic, coh, np.asarray(coh).reshape((self.Vdim,self.Vdim))
            
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Equlibrium Density Matrix
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def EqulibriumDensityMatrix_Scraped(self,H,T,HT_approx):
        """
        Equlibrium Density Matrix
        INPUT
        -----
        H         : Hamiltonian always in energy unit and not in frequency unit.
        T         : Temperature
        HT_approx : if True, high temperature approximation will be considered
        
        OUTPUT     
        ------
        Equlibrium Density Matrix      
        """
        
        rho_T = np.zeros((self.Vdim,self.Vdim))
        if HT_approx: 
            E = np.eye(self.Vdim)   
            rho_T = (E - H/(self.kb*T))/np.trace(E - H/(self.kb*T)) # High Temperature Approximation
        else:
            rho_T = expm(-H/(self.kb*T))/np.trace(expm(-H/(self.kb*T))) # General
            
        print("Trace of density metrix = ", (np.trace(rho_T)).real)    

        return rho_T    

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
        
        LarmorF = self.LarmorF
        Sz = self.Sz_

        rho_T = np.zeros((self.Vdim,self.Vdim))
        
        H_Eq_T = np.zeros((self.Vdim,self.Vdim))
        
        for i in range(self.Nspins):
            H_Eq_T = H_Eq_T + self.Convert_FreqUnitsTOEnergy(LarmorF[i] * Sz[i]) / (self.kb*Ti[i])  
         
        if HT_approx:
            E = np.eye(self.Vdim)
            rho_T = (E - H_Eq_T)/np.trace(E - H_Eq_T) # High Temperature Approximation
        else:    
            rho_T = expm(-H_Eq_T)/np.trace(expm(-H_Eq_T)) # General
            
        print("Trace of density metrix = ", (np.trace(rho_T)).real)    

        return rho_T 
        
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

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Commutators and Superoperators
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Commutator(self,A,B):
        """
        Commutator
        INPUT
        -----
        A : matrix A
        B : matrix B

        OUTPUT     
        ------
        Commutator [A,B]      
        """     
        return np.matmul(A,B) - np.matmul(B,A)
    
    def DoubleCommutator(self,A,B,rho):
        """
        Double Commutator 
        INPUT
        -----
        A   : matrix A
        B   : matrix B
        rho : matrix rho

        OUTPUT     
        ------
        Double Commutator [A,[B,rho]]      
        """     
        C = self.Commutator(B,rho)
        return self.Commutator(A,C)
    
    def AntiCommutator(self,A,B):
        """
        Anti Commutator
        INPUT
        -----
        A   : matrix A
        B   : matrix B

        OUTPUT     
        ------
        Anti Commutator {A,B}      
        """     
        return np.matmul(A,B) + np.matmul(B,A)      
        
    def CommutationSuperoperator(self,X):
        """
        Commutation Superoperator [H,rho] = left(H) [rho] - right(H) [rho] = H rho - rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        Commutation Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        if self.SparseM:
            return sparse.csc_matrix(np.kron(X,Id) - np.kron(Id,X.T))
        else:
            return np.kron(X,Id) - np.kron(Id,X.T)

    def AntiCommutationSuperoperator(self,X):
        """
        Anti Commutation Superoperator: {H,rho} = left(H) [rho] + right(H) [rho] = H rho + rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        anti Commutation Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        if self.SparseM:
            return sparse.csc_matrix(np.kron(X,Id) + np.kron(Id,X.T))
        else:
            return np.kron(X,Id) + np.kron(Id,X.T)

    def Left_Superoperator(self,X):
        """
        Left Superoperator: left(H) [rho] = H rho
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        left Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        if self.SparseM:
            return sparse.csc_matrix(np.kron(X,Id))
        else:
            return np.kron(X,Id)
        
    def Right_Superoperator(self,X):
        """
        Right Superoperator: right(H) [rho] = rho H
        
        INPUT
        -----
        X : matrix X

        OUTPUT     
        ------
        right Superoperator   
        """     
        Id = np.identity((X.shape[-1]))
        if self.SparseM:
            return sparse.csc_matrix(np.kron(Id,X.T))
        else:
            return np.kron(Id,X.T)        
    
    def DoubleCommutationSuperoperator(self,X,Y):
        """
        Double Commutation Superoperator
        INPUT
        -----
        X : matrix X
        Y : matrix Y

        OUTPUT     
        ------
        Double Commutation Superoperator  
        """     
        Idx = np.identity((X.shape[-1]))
        Idy = np.identity((Y.shape[-1]))
        if self.SparseM:
            return sparse.csc_matrix(np.matmul(np.kron(X,Idx) - np.kron(Idx,X.T), np.kron(Y,Idy) - np.kron(Idy,Y.T)))
        else:
            return np.matmul(np.kron(X,Idx) - np.kron(Idx,X.T), np.kron(Y,Idy) - np.kron(Idy,Y.T)) 
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotation (Pulse) in Hilbert space and Liouville Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    def Pulse_Phase(self,Sx,Sy,phase):
        """
        Pulse with defined phase; cos(phase) Sx + Sin(phase) Sy
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Phase: Phase in deg
        
        OUTPUT
        ------        
        return spin operator about which to rotate spin
        """
        phase = np.pi * phase / 180.0
        return np.cos(phase) * np.sum(Sx,axis=0) + np.sin(phase) * np.sum(Sy,axis=0)
    
    def Receiver_Phase(self,Sx,Sy,phase): 
        """
        Detection operator with phase; (Sx + 1j Sy) * exp(1j phase)
        
        INPUT
        -----
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Phase: Phase in deg
                
        OUTPUT
        ------        
        return detection operator rotated by reciever phase
        """
        phase = np.pi * phase / 180.0
        return (np.sum(Sx,axis=0) + 1j * np.sum(Sy,axis=0)) * np.exp(1j * phase)        
    
    def Rotation_CyclicPermutation(self, A, B, theta):
    
        """
        Rotation of an operator, when the operator and spin operator follw the relation
        [A,B] = j C (Cylic Commutation Relation)
        
        INPUT
        -----
        A      : Operator about which rotation happens
        B      : Operator to rotate
        theta  : angle in radian
        
        OUTPUT
        ------
        EXP(-j A * theta) @ B @ EXP(j A * theta) = B cos(theta) - j [A, B] sin(theta) = B cos(theta) + C sin(theta)
        """
        
        if A == B:
            Bp = B
        else:
            Bp = B * np.cos(np.pi*theta/180.0) - 1j * self.Commutator(A,B) * np.sin(np.pi*theta/180.0)
            
        return Bp
           
    def Rotate_Pulse(self,rho,theta_rad,operator):
        if self.PropagationSpace == "Hilbert": 
            """
            Rotation in Hilbert Space
            INPUT
            -----
            rho       : intial density matrix or operator (eg: hamiltonian)
            theta_rad : Angle to be rotated in degree
            operator  : Spin Operator for rotation

            OUTPUT     
            ------
            rho       : Rotated density matrix or operator (eg: hamiltonian)       
            """     
            theta_rad = np.pi * theta_rad / 180.0
            U = expm(-1j * theta_rad * operator)
            return self.Matrix_Tol(np.matmul(U,np.matmul(rho,U.T.conj())))  

        if self.PropagationSpace == "Liouville":
            """
            Rotation in Liouville Space
            INPUT
            -----
            Lrho      : intial state
            theta_rad : Angle to be rotated in degree
            operator  : Spin Superoperator for rotation

            OUTPUT     
            ------
            Lrho       : final state      
            """     

            theta_rad = np.pi * theta_rad / 180.0
            return self.Matrix_Tol(expm(-1j * theta_rad * self.CommutationSuperoperator(operator)) @ rho) 
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Liouville Vectors
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Vector_L(self,X):
        """
        Liouville Vector: Vectorize the operator
        INPUT
        -----
        X : Operator to be vectorized. eg: density matrix

        OUTPUT     
        ------
        Vectorized operator      
        """     
        dim = self.Vdim
        return np.reshape(X,(dim**2,-1))
    
    def Detection_L(self,X):
        """
        Liouville Vector for detection: Vectorize the operator
        INPUT
        -----
        X : Operator to be vectorized. eg: Sz

        OUTPUT     
        ------
        Vectorized operator for detection     
        """     
        X = self.Vector_L(X)
        return X.conj().T
        
    def ProductOperators_ConvertToLiouville(self,Basis_X):
        """
        Convert productor operator basis in Himlbert space to Liouville Space
        
        INPUT
        -----
        Basis_X: Product operator basis in Hilbert Space
        
        OUTPUT
        ------
        return Product operator basis in liouville Space
        """ 
        
        dim = len(Basis_X)  
        Basis_out = []
        
        for i in range(dim): 
            Basis_out.append(self.Vector_L(np.asarray(Basis_X[i])))
            
        return Basis_out  
        
    def Liouville_Bracket(self,A,B,C):
        """
        Liouville Bracket
        """
        
        return np.trace(self.Adjoint(A) @ B @ C).real       
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Probability Desnity Function
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    def PDFgaussian(self, x, std, mean):  
        """
        Probabilty Distribution Function Gaussian
        
        INPUT
        -----
        x: array of variable for Gaussian Probabilty Distribution Function
        std: standard deviation
        mean: mean
        
        OUTPUT
        ------        
        return normalized Gaussian Probabilty Distribution Function
        """
        gaussian =  (1/np.sqrt(2*np.pi*std**2)) * np.exp(-1*(x-mean)**2/(2*std**2))
        return gaussian/np.sum(gaussian)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Matrix Visualization
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
    def MatrixPlot(self,fig_no,M,xlabel,ylabel):
        """
        Matrix Plotting
        
        INPUT
        -----
        fig_no: figure number
        M: Matrix
        OUTPUT
        ------        
        return plot matrix
        """
        
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        labelx = xlabel
        labely = ylabel   
                                
        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize
        
        fig = plt.figure(fig_no)
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(M, interpolation='nearest',cmap=cmap[1],vmax=abs(M).max(), vmin=-abs(M).max())
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(labelx)))
        ax.set_yticks(np.arange(len(labely)))
        ax.set_xticklabels(labelx,rotation='vertical')
        ax.set_yticklabels(labely) 
        plt.tight_layout()
        plt.show()

    def MatrixPlot_slider(self,fig_no,t,rho_t,xlabel,ylabel):
        """
        Matrix Plotting as function of time
        
        INPUT
        -----
        fig_no: figure number
        t: Time array
        rho_t: array of density matrices for each time 
        OUTPUT
        ------        
        Plot matrix with slider, move slider to see density matrix at different time.
        """
        
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        labelx = xlabel
        labely = ylabel           
                    
        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize
        plt.rcParams["figure.autolayout"] = True
        
        fig = plt.figure(fig_no)
        ax = fig.add_subplot(111)
        X = ax.matshow(rho_t[0].real, interpolation='nearest',cmap=cmap[1])
        
        cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.65])
        cbar = fig.colorbar(X, cax = cbaxes)
        
        ax.set_title('T=%2.3f'%t[0])
        ax.set_xticklabels([''] + labelx,fontsize=self.PlotFontSize)
        ax.set_yticklabels([''] + labely,fontsize=self.PlotFontSize) 
        
        fig.subplots_adjust(left=0.25, bottom=0.25)
        axfreq = fig.add_axes([0.2, 0.001, 0.65, 0.03])
        index_slider = Slider(ax=axfreq,label='index',valmin=0,valmax=t.shape[-1], valinit=0)
        
        def update(val):
            X = ax.matshow(rho_t[int(index_slider.val)].real, interpolation='nearest',cmap=cmap[1])
            ax.set_title('T=%2.3f'%t[int(index_slider.val)])
            cbar.update_normal(X)
            fig.canvas.draw_idle()
            
        index_slider.on_changed(update)
        
        plt.show()
        
    def MatrixPlot3D(self,fig_no,rho,xlabel,ylabel):
        """
        Matrix Plot 3D
        
        INPUT
        -----
        fig_no: Figure number
        rho: density matrix
        OUTPUT
        ------        
        Plot 3D matrix
        """
             
        labelx = xlabel
        labely = ylabel

        plt.rcParams['figure.figsize'] = self.PlotFigureSize
        plt.rcParams['font.size'] = self.PlotFontSize
                        
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True)
        ax1 = plt.axes(projection = "3d")
        
        numofCol = rho.shape[-1]
        numofRow = rho.shape[0]
        
        xpos = np.arange(0, numofCol, 1)
        ypos = np.arange(0, numofRow, 1)
        xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
        
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(numofRow*numofRow)
        
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = rho.flatten()
        
        positive = dz.copy()
        negative = dz.copy()
        positive[positive<0] = 0
        negative[negative>=0] = 0
                
        ax1.bar3d(xpos,ypos,zpos, dx, dy, dz, color='b', alpha=0.5)

        #ax1.bar3d(xpos,ypos,zpos, dx, dy, positive, color='b', alpha=0.5)
        #ax1.bar3d(xpos,ypos,zpos, dx, dy, -negative, color='r', alpha=0.5)
        
        ticksx = np.arange(0.5, rho.shape[-1], 1)
        ticksy = np.arange(0.6, rho.shape[-1], 1)
        #ax1.set_xticklabels(label)
        #ax1.set_yticklabels(label)
        plt.xticks(ticksx,labelx,fontsize=self.PlotFontSize)
        plt.yticks(ticksy,labely,fontsize=self.PlotFontSize)
        ax1.set_zlim(np.min(rho),np.max(rho))
        #ax1.set_zlim(0,np.max(rho))
        ax1.grid(False)
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plotting and Fourier transform
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    
    def Plotting(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        fig_no    : figure number
        x         : x array (Horizontal axis)
        y         : y array (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : colour of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(x,y,linewidth=3.0,color=col)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        xli,xlf = self.PlotXlimt
        yli,ylf = self.PlotYlimt
        ax1.set_xlim(xli,xlf)
        ax1.set_ylim(yli,ylf)
        plt.show()
        
    def Plotting_SpanSelector(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        fig_no    : figure number
        x         : x array (Horizontal axis)
        y         : y array (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : colour of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])

        ax1.plot(x,y,linewidth=3.0,color=col)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        xli,xlf = self.PlotXlimt
        yli,ylf = self.PlotYlimt
        ax1.set_xlim(xli,xlf)
        ax1.set_ylim(yli,ylf)
        
        vline_left = ax1.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax1.axvline(0, color='red', linestyle='--', visible=False)
        
        span_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, fontsize=self.PlotFontSize, verticalalignment='top')
        def onselect(xmin, xmax):
            # Update the vertical lines' positions
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
                
            # Update the text annotation with the selected span
            span_text.set_text(f"Selected Span = {xmax-xmin:.4f}")
    
            # Redraw the canvas to show updates
            fig.canvas.draw_idle()  
            
        span_selector = SpanSelector(ax1, onselect, direction='horizontal', useblit=True)           
        
        return fig,span_selector        

    def PlottingTwin(self,fig_no,x,y1,y2,xlab,ylab1,ylab2,col1,col2):
        """
        Plotting Twin Axis (y)
        
        INPUT
        -----
        fig_no: figure number
        x: x array
        y1: y1 array
        y2: y2 array
        xlabel: x label
        ylab1: y1 label
        ylab2: y2 label
        col1: color for y1
        col2: color for y2
        
        OUTPUT
        ------        
        plot
        """
        
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
	    
        ax1.plot(x,y1,linewidth=3.0,color=col1)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab1, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)

        ax10 = ax1.twinx()
        ax10.plot(x,y2,linewidth=3.0,color=col2)

        ax10.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax10.set_ylabel(ylab2, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax10.legend(fontsize=self.PlotFontSize,frameon=False)
        ax10.tick_params(axis='both',labelsize=14)
        ax10.grid(True, linestyle='-.')
	    #plt.savefig('figure.pdf',bbox_inches='tight')
        plt.show()

    def PlottingTwin_SpanSelector(self,fig_no,x,y1,y2,xlab,ylab1,ylab2,col1,col2):
        """
        Plotting Twin Axis (y)
        
        INPUT
        -----
        fig_no: figure number
        x: x array
        y1: y1 array
        y2: y2 array
        xlabel: x label
        ylab1: y1 label
        ylab2: y2 label
        col1: color for y1
        col2: color for y2
        
        OUTPUT
        ------        
        plot
        """
        
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
	    
        ax1.plot(x,y1,linewidth=3.0,color=col1)

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab1, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)

        ax10 = ax1.twinx()
        ax10.plot(x,y2,linewidth=3.0,color=col2)

        ax10.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax10.set_ylabel(ylab2, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax10.legend(fontsize=self.PlotFontSize,frameon=False)
        ax10.tick_params(axis='both',labelsize=14)
        ax10.grid(True, linestyle='-.')
	    #plt.savefig('figure.pdf',bbox_inches='tight')

        vline_left = ax10.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax10.axvline(0, color='red', linestyle='--', visible=False)
        
        span_text = ax10.text(0.05, 0.95, "", transform=ax10.transAxes, fontsize=self.PlotFontSize, verticalalignment='top')
        def onselect(xmin, xmax):
            # Update the vertical lines' positions
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
                
            # Update the text annotation with the selected span
            span_text.set_text(f"Selected Span = {xmax-xmin:.2f}")
    
            # Redraw the canvas to show updates
            fig.canvas.draw_idle()  
            
        span_selector = SpanSelector(ax10, onselect, direction='horizontal', useblit=True)           
        
        return fig,span_selector

    def PlottingMulti(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        figure    : figure number
        x         : [x1 array, x1 array, ...] (Horizontal axis)
        y         : [y1 array, y1 array, ... ] (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : [colour 1, colour 2, ... ] of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        
        for i in range(len(x)):
            ax1.plot(x[i],y[i],linewidth=3.0,color=col[i])

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)
        plt.show()

    def PlottingMulti_SpanSelector(self,fig_no,x,y,xlab,ylab,col):
        """
        Plotting the signal
        INPUT
        -----
        figure    : figure number
        x         : [x1 array, x1 array, ...] (Horizontal axis)
        y         : [y1 array, y1 array, ... ] (Vertical axis)
        xlab      : x label
        ylab      : y label
        col       : [colour 1, colour 2, ... ] of the plot

        OUTPUT     
        ------
        plot      
        """     
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)

        ax1 = fig.add_subplot(spec[0, 0])
        
        for i in range(len(x)):
            ax1.plot(x[i],y[i],linewidth=3.0,color=col[i])

        ax1.set_xlabel(xlab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        ax1.set_ylabel(ylab, fontsize=self.PlotFontSize, color='black',fontweight='bold')
        #ax1.legend(fontsize=self.PlotFontSize,frameon=False)
        ax1.tick_params(axis='both',labelsize=14)
        ax1.grid(True, linestyle='-.')
        #ax1.set_xlim(xli,xlf)

        vline_left = ax1.axvline(0, color='red', linestyle='--', visible=False)
        vline_right = ax1.axvline(0, color='red', linestyle='--', visible=False)
        
        span_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, fontsize=self.PlotFontSize, verticalalignment='top')
        def onselect(xmin, xmax):
            # Update the vertical lines' positions
            vline_left.set_xdata([xmin])
            vline_right.set_xdata([xmax])
            vline_left.set_visible(True)
            vline_right.set_visible(True)
                
            # Update the text annotation with the selected span
            span_text.set_text(f"Selected Span = {xmax-xmin:.2f}")
    
            # Redraw the canvas to show updates
            fig.canvas.draw_idle()  
            
        span_selector = SpanSelector(ax1, onselect, direction='horizontal', useblit=True)           
        
        return fig,span_selector

    def Plotting3DWire(self,fig_no,x,y,z,xlab,ylab,title,upL,loL):
        """
        Plot 3D Surface
        
        INPUT
        -----
        fig_no: Figure number
        x: x data
        y: y data
        z: z data, function of x,y
        xlab: x label
        ylab: y label
        title: Title of the plot
        upL: Upper limit of X and Y axis
        loL: Lower limit of X and Y axis        
        OUTPUT
        ------        
        return the wire plot
        """
        rc('font', weight='bold')
        #ax = plt.figure(fig_no,figsize=(10, 5)).add_subplot(projection='3d')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(spec[0, 0],projection='3d')
        
        x1 = x.copy()
        y1 = y.copy()
        x1[x1>upL] = np.nan
        y1[y1>upL] = np.nan
        x1[x1<loL] = np.nan
        y1[y1<loL] = np.nan        
        
        X,Y = np.meshgrid(x1,y1)
        wire = ax1.plot_wireframe(X, Y, z, lw=0.5, rstride=8, cstride=8) #, alpha=0.3
        # rstride=0 for row stride set to 0
        # ctride=0 for column stride set to 0
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        ax1.set_xlim3d(loL,upL)
        ax1.set_ylim3d(loL,upL)
        plt.show()
                
    def PlottingContour(self, fig_no,x,y,z,xlab,ylab,title):
        """
        Plot Contour
        
        INPUT
        -----
        fig_no: Figure number
        x: x data
        y: y data
        z: z data, function of x,y
        xlab: x label
        ylab: y label
        titile: Title of the plot
        
        OUTPUT
        ------        
        return the contour plot
        """
        cmap = [cm.RdBu, cm.seismic, cm.bwr, cm.RdGy]
        rc('font', weight='bold')
        fig = plt.figure(fig_no,constrained_layout=True, figsize=self.PlotFigureSize)
        spec = fig.add_gridspec(1, 1)
        
        ax1 = fig.add_subplot(spec[0, 0])
        plotC = ax1.contour(z, 10, extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap[1], vmax=abs(z).max(), vmin=-abs(z).max()) 
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        ax1.set_title(title)
        cbar = fig.colorbar(plotC)
        plt.show()
        
    def PlottingSphere(self, fig_no,Mx,My,Mz,rho_eq,Sz,plot_vector,scale_datapoints):
        """
        Plotting magnetization evolution in a unit sphere
        
        INPUT
        -----
        fig_no: Figure number
        Mx: Array of Mx
        My: Array of My
        Mz: Array of Mz
        rho_eq: equlibrium density matrix
        Sz: Spin operator Sz
        plot_vector: If True, vector will be plotted
        scale_datapoints: scale points in the Mx, My and Mz; Mx[::scale_datapoints]
        
        OUTPUT
        ------        
        return sphere plot
        """        
        
        sphera_radius = self.InnerProduct(Sz,rho_eq)
        
        # Create a sphere
        phi, theta = np.mgrid[0.0:2.0 * np.pi:100j, 0.0:np.pi:50j]
        x = sphera_radius * np.sin(theta) * np.cos(phi)
        y = sphera_radius * np.sin(theta) * np.sin(phi)
        z = sphera_radius * np.cos(theta)
                
        fig = plt.figure(fig_no,figsize=self.PlotFigureSize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, color='c', alpha=0.3, rstride=5, cstride=5, linewidth=0.5, edgecolor='k')
            
        if plot_vector:
            for mx,my,mz in zip(Mx,My,Mz):  
                ax.quiver(0, 0, 0, mx, my, mz, color='r', arrow_length_ratio=0.1) 
                
        ax.plot(Mx[::scale_datapoints], My[::scale_datapoints], Mz[::scale_datapoints], color='b', linewidth=2)
        ax.quiver(0, 0, 0, Mx[0], My[0], Mz[0], color='r', arrow_length_ratio=0.1)  
        ax.quiver(0, 0, 0, Mx[-1], My[-1], Mz[-1], color='b', arrow_length_ratio=0.1)
        ax.view_init(10, 20)
        ax.set_xlabel('Mx')
        ax.set_ylabel('My')
        ax.set_zlabel('Mz')     
        plt.show()  
        
    def PlottingMultimodeAnalyzer(self,t,freq,sig,spec):
        """
        Multimode Analyzer
        
        INPUT
        -----
        t: time
        freq: frequency
        sig: signal or FID
        spec: spectrum
        
        OUTPUT
        ------
        plot 4 figures 
        Figure 1,1 Signal
        Figure 1,2 Spectrum
        Figure 2,1 Signal
        Figure 2,2 Spectrum
        """            
        rc('font', weight='bold')
        fig, ax = plt.subplots(2,2,figsize=self.PlotFigureSize)


        line1, = ax[0,0].plot(t,sig,"-", color='green')
        ax[0,0].set_xlabel("time [s]")
        ax[0,0].set_ylabel("signal" )
        ax[0,0].grid()

        vline1 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
        vline2 = ax[0,1].axvline(color='k', lw=0.8, ls='--')
        text1 = ax[0,1].text(0.0, 0.0, '', transform=ax[0,1].transAxes)
        line2, = ax[0,1].plot(freq,spec,"-", color='green')
        ax[0,1].set_xlabel("Frequency [Hz]")
        ax[0,1].set_ylabel("spectrum" )
        #ax[0,1].set_xlim(-40,40)
        ax[0,1].grid()

        line3, = ax[1,0].plot(freq,spec,"-", color='green')
        ax[1,0].set_xlabel("Frequency [Hz]")
        ax[1,0].set_ylabel("spectrum" )
        #ax[1,0].set_xlim(-40,40)
        ax[1,0].grid()

        vline3 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
        vline4 = ax[1,1].axvline(color='k', lw=0.8, ls='--')
        text2 = ax[1,1].text(0.0, 0.0, '', transform=ax[1,1].transAxes)
        line4, = ax[1,1].plot(t,sig,"-", color='green')
        ax[1,1].set_xlabel("time [s]")
        ax[1,1].set_ylabel("signal" )
        ax[1,1].grid()
        #plt.savefig(folder + '/pic3.pdf',bbox_inches='tight')

        fourier = Fanalyzer(sig.real,sig.imag,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2)
        fig.canvas.mpl_connect("button_press_event",fourier.button_press)
        fig.canvas.mpl_connect("button_release_event",fourier.button_release)
        
        return fig,fourier
                
    def WindowFunction(self,t,signal,LB):
        """
        Induce signal decay
        INPUT
        -----
        t      : time array
        signal : signal array
        LB     : decay rate

        OUTPUT     
        ------
        decaying signal      
        """     
        window = np.exp(-LB*t)
        return signal*window
    
    def FourierTransform(self,signal,fs,zeropoints):
        """
        Fourier Transform
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        Fourier transform      
        """     
        signal[0] = signal[0]
        spectrum = np.fft.fft(signal,zeropoints*signal.shape[-1])
        spectrum = np.fft.fftshift(spectrum)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[-1])
        return freq, spectrum  
        
    def PhaseAdjust_PH0(self,spectrum,PH0):
        """
        Phase adjust PH0
        
        INPUT
        -----
        spectrum: spectrum to phase
        PH0: Phase
        
        OUTPUT
        ------        
        return phased spectrum
        """
        
        return spectrum * np.exp(1j * 2 * np.pi * PH0 / 180.0)

    def PhaseAdjust_PH1(self,freq,spectrum,pivot,slope):
        """
        Phase adjust PH0
        
        INPUT
        -----
        freq: frequency axis
        spectrum: spectrum to phase
        pivot: frequency where first order pahse correction is zero
        slope: rate of change of phase
        
        OUTPUT
        ------        
        return phased spectrum
        """
        
        freq_axis = np.arange(len(freq))
        pivot_corrd = np.searchsorted(freq, pivot)
        PH1 = slope * -1.0e-3 * (freq_axis - freq_axis[pivot_corrd])
        return spectrum * np.exp(1j * 2 * np.pi * PH1 / 180.0)

    def FourierTransform2D(self,signal,fs1,fs2,zeropoints):
        """
        Fourier Transform 2D
        
        INPUT
        -----
        signa: signal array
        fs1: sampling rate (Indirect Dimension)
        fs2: sampling rate (Direct Dimension)
        zeropoints: zero filling (zeropoints * Npoints)
        
        OUTPUT
        ------        
        
        """     
        signal[:,0] = signal[:,0]/2
        spectrum = np.fft.fft2(signal,(zeropoints*signal[:,0].shape[-1],zeropoints*signal[0,:].shape[-1]),(1,0))
        spectrum = np.fft.fftshift(spectrum)
        freq1 = np.linspace(-fs1/2,fs1/2,spectrum.shape[-1])
        freq2 = np.linspace(-fs2/2,fs2/2,spectrum.shape[0])
        return freq1, freq2, spectrum

    def FourierTransform2D_F1(self,signal,fs,zeropoints):
        """
        Fourier Transform 1D - F1 (Indirect Dimension)
        
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        return frequency and Fourier transform      
        """
        spectrum = np.zeros((signal.shape[0],signal.shape[-1]),dtype=np.cdouble)
        for i in range(signal.shape[-1]):
            spec = np.fft.fft(signal[:,i])
            spectrum[:,i] = np.fft.fftshift(spec)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[0])
        return freq, spectrum

    def FourierTransform2D_F2(self,signal,fs,zeropoints):
        """
        Fourier Transform 1D (Direct Dimension)
        INPUT
        -----
        signal      : signal array
        fs          : sampling rate (half of the bandwidth)
        zeropoints  : zero filling (zeropoints * Npoints)

        OUTPUT     
        ------
        return frequency and Fourier transform      
        """     
        signal[0] = signal[0]/2
        spectrum = np.fft.fft(signal,zeropoints*signal.shape[-1],1)
        spectrum = np.fft.fftshift(spectrum)
        freq = np.linspace(-fs/2,fs/2,spectrum.shape[-1])
        return freq, spectrum

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Coherence Filter
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    def Filter_T00(self, rho, index):
        """"
        T00 Filter  (density matrix in Zeeman basis)
        
        INPUT
        -----
        rho: input density matrix
        index: index of spins
        Sx: Spin operator Sx
        Sy: Spin operator Sy
        Sz: Spin operator Sz
        
        OUTPUT
        ------
        return density matrix woth only zero quantum coherence between transition corresponds to spins with index, index[0] and index[1], Ix Sx + Iy Sy + Iz Sz
        """

        Sx = self.Sx_
        Sy = self.Sy_ 
        Sz = self.Sz_
        
        ZQx =  Sx[index[0]] @ Sx[index[1]] + Sy[index[0]] @ Sy[index[1]] + Sz[index[0]] @ Sz[index[1]] 
        ZQy = Sy[index[0]] @ Sx[index[1]] - Sx[index[0]] @ Sy[index[1]]
        Filter_T00 = ZQx
        Filter_T00[Filter_T00 == 0.5] = 1
        Filter_T00[Filter_T00 == -0.5] = 1
        Filter_T00[Filter_T00 == 0.25] = 1
        Filter_T00[Filter_T00 == -0.25] = 1
        return Filter_T00, np.multiply(rho,Filter_T00) 
        
    def Filter_Coherence(self,rho,Allow_Coh,Sz,Hz_lab):  
        """
        Filter allow only selected coherence (density matrix in Zeeman basis)
        
        INPUT
        -----
        rho:
        Allow_Coh:
        Sz:
        Hz_lab:
        
        OUTPUT
        ------
        return density matrix with selected coherence order 
        """  
        
        Basis_Zeeman, dic_Zeeman, coh_Zeeman, coh_Zeeman_array = self.ProductOperators_Zeeman(Sz,Hz_lab)
        Max_Coh = int(np.max(coh_Zeeman_array))
        coh_Zeeman_array[coh_Zeeman_array == Allow_Coh] = 1000
        for i in range(Max_Coh + 1):
            if i == 0:
                coh_Zeeman_array[coh_Zeeman_array == i] = 0
            else:
                coh_Zeeman_array[coh_Zeeman_array == i] = 0 
                coh_Zeeman_array[coh_Zeeman_array == -i] = 0   
        coh_Zeeman_array = coh_Zeeman_array / 1000.0
        
        return coh_Zeeman_array, np.multiply(rho,coh_Zeeman_array)
        
                        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Time evolution of Density Matrix in Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
        
    def ShapedPulse_H(self,t):
        """
        """
        
        if self.ShapeFunc == "Off Resonance":
            return self.Zeeman_B1_Offresonance(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)
        if self.ShapeFunc == "Bruker":
            return self.Zeeman_B1_ShapedPulse(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)        
                
    def Evolution(self,rhoeq,rho,Hamiltonian,Relaxation=None):

        Pmethod = self.PropagationMethod
        ode_method = self.OdeMethod
        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        Sx = self.Sx_
        Sy = self.Sy_ 
        Sz = self.Sz_
        Sp = self.Sp_
        Sm = self.Sm_     

        if self.PropagationSpace == "Hilbert":
            """
            Evolution of density matrix
            INPUT
            -----
            rho         : intial state
            Hamiltonian : Hamiltonian of evolution
            detection   : detection operator
            dt          : time step
            Npoints     : number of time points
            method      : "unitary propagator"  Propagate the hamiltonian by unitary matrix (exp(-j H dt))
                        : "solve ivp" solve the Liouville with differential equation solver (radiation damping and relaxation included)
            Rprocess    :  "No Relaxation" 
                        or "Phenomenological"
                        or "Auto-correlated Random Field Fluctuation" 
                        or "Auto-correlated Dipolar Heteronuclear"
                        or "Auto-correlated Dipolar Homonuclear"
            
            OUTPUT     
            ------
            t       : time
            rho     : Array of density matrix      
            """ 
            
            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt)
                for i in range(Npoints):
                    rho = np.matmul(U,np.matmul(rho,U.T.conj()))
                    rho_t[i] = rho
                    
            if Pmethod == "ODE Solver":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)                       
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rso_temp = self.Relaxation(rho_temp-rhoeq)
                    H = Hamiltonian      
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho	            

            if Pmethod == "ODE Solver ShapedPulse":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)                       
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rso_temp = self.Relaxation(rho_temp-rhoeq)
                    H_shapePulse = self.ShapedPulse_H(t)
                    H = H_shapePulse + Hamiltonian
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho

            if Pmethod == "ODE Solver Relaxation and Phenomenological":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)                       
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rprocess2 = "Phenomenological Input"
                    Rso_temp = self.Relaxation(rho_temp-rhoeq) + self.Relaxation(rho_temp-rhoeq)
                    H = Hamiltonian    
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho

            if Pmethod == "ODE Solver Stiff RealIntegrator": 
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: Nill
                Remarks: 
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)                       
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                rhoi = (rho.reshape(-1))
                rho_RI = np.zeros((2*rhoi.shape[-1]))
                rho_RI[0::2] = rhoi.real
                rho_RI[1::2] = rhoi.imag
                
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho = np.reshape(rho[0::2] + 1j * rho[1::2],(self.Vdim,self.Vdim))
                    rhodot = np.zeros((2*rhoi.shape[-1]))
                    Rso = self.Relaxation(rho-rhoeq)
                    H = Hamiltonian       
                    rhodot[0::2] = (-1j * self.Commutator(H,rho) - Rso).reshape(-1).real  
                    rhodot[1::2] = (-1j * self.Commutator(H,rho) - Rso).reshape(-1).imag     
                    return rhodot 
                
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rho_RI,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)

                t, rho2d = rhoSol.t, rhoSol.y
                rho2d_R =  rho2d[0::2]
                rho2d_I =  rho2d[1::2]
                
                for i in range(Npoints):          
                    rho_R = np.reshape(rho2d_R[:,i],(self.Vdim,self.Vdim))
                    rho_I = np.reshape(rho2d_I[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho_R + 1j * rho_I
                                                                                
            return t, rho_t
        
        if self.PropagationSpace == "Liouville":
            """
            Evolution of density vector
            INPUT
            -----
            Lrho         : intial state vector
            Lrhoeq       : equlibrium state vector
            LHamiltonian : Hamiltonian of evolution
            RsuperOP     : Relaxation Superoperator
            dt          : time step
            Npoints     : number of time points
            method      : "unitary propagator"  Propagate the hamiltonian by unitary matrix (exp(-j H dt))
                        "Relaxation"          Propagate the hamiltonian by unitary matrix with relaxation included
                        : "solve ivp" solve the Liouville with differential equation solver (relaxation included)

            OUTPUT     
            ------
            t       : time
            Lrho     : array of final density state vector     
            """  

            Sx = self.Sx_
            Sy = self.Sy_

            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt)
                for i in range(Npoints):
                    rho = np.matmul(U,rho)  
                    rho_t[i] = rho  

            if Pmethod == "Unitary Propagator Sparse":  
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = sparse.linalg.expm(-1j * Hamiltonian * dt) # LHamiltonian is sparse matrix
                for i in range(Npoints):
                    rho = U.dot(rho)  
                    rho_t[i] = rho
            
            if Pmethod == "Relaxation":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                for i in range(Npoints):
                    rho = np.matmul(U,rho - rhoeq) + rhoeq
                    rho_t[i] = rho        

            if Pmethod == "Relaxation Sparse":   
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = sparse.linalg.expm(-1j * Hamiltonian * dt - Relaxation * dt) # LHamiltonian and RsuperOP are sparse matrix           
                for i in range(Npoints):
                    rho = U.dot(rho - rhoeq) + rhoeq
                    rho_t[i] = rho

            if Pmethod == "Relaxation Lindblad":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                for i in range(Npoints):
                    rho = np.matmul(U,rho)
                    rho_t[i] = rho 

            if Pmethod == "Relaxation Lindblad Sparse":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = sparse.linalg.expm(-1j * Hamiltonian * dt - Relaxation * dt) # LHamiltonian and RsuperOP are sparse matrix
                for i in range(Npoints):
                    rho = np.matmul(U,rho)
                    rho_t[i] = rho 

            if Pmethod == "ODE Solver":
                """
                Reference: Equation 47, A liouville space formulation of wangsness-bloch-redfield theory of nuclear spin relaxation suitable for machine computation. I. fundamental aspects, Slawomir Szymanski et.al., https://doi.org/10.1016/0022-2364(86)90334-3
                """
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex) 
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                Lrho = np.reshape(rho,rho.shape[0]) + 0 * 1j            
                Lrhoeq = np.reshape(rhoeq,rhoeq.shape[0])
                
                def rhoDOT(t,Lrho,LHamiltonian,RsuperOP,Lrhoeq,Sx,Sy):
                    LH = LHamiltonian
                    rhodot = np.zeros((self.Ldim),dtype=complex)
                    rhodot = -1j * np.matmul(LH,Lrho) - np.matmul(RsuperOP,Lrho-Lrhoeq)
                    rhodot = np.reshape(rhodot,rhodot.shape[0])
                    return rhodot
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],Lrho,method=ode_method,t_eval=t,args=(Hamiltonian,Relaxation,Lrhoeq,Sx,Sy), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, rho_sol = rhoSol.t, rhoSol.y
                print(rho_sol.shape)
                for i in range(Npoints):
                    rho_t[i] = np.reshape(rho_sol[:,i],(self.Ldim,1))
                                        
            return t, rho_t 
            
    def Expectation(self,rho_t,detection):

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)
    
        if self.PropagationSpace == "Hilbert":
            """
            Expectation Value
            
            INPUT
            -----
            rho_t: array of 2d matrix, the density matrix
            detection: observable
            dt: dwell time
            Npoints: Acquisition points 
            
            
            OUTPUT
            ------        
            t: array, Time
            signal: array, Expectation values
            """

            signal = np.zeros(Npoints,dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            for i in range(Npoints):
                #signal[i] = np.trace(np.matmul(detection,rho_t[i]))
                signal[i] = np.trace(np.matmul(rho_t[i],detection))
            return t, signal 

        if self.PropagationSpace == "Liouville":
            """
            Expectation Value
            
            INPUT
            -----
            Lrho_t: array of coloumn Vectors, the density matrix
            Ldetection: observable
            dt: dwell time
            Npoints: Acquisition points 
            
            
            OUTPUT
            ------        
            t: array, Time
            signal: array, Expectation values
            """
            
            signal = np.zeros(Npoints,dtype=complex)
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            for i in range(Npoints):
                signal[i] = np.trace(detection @ rho_t[i])
            return t, signal   
       
    
    def Convert_LrhoTO2Drho(self,Lrho):
        """
        Convert a Vector into a 2d Matrix
        
        INPUT
        -----
        Lrho: density matrix, coloumn vector
        OUTPUT
        ------        
        return density matrix, 2d array
        """
        
        return np.reshape(Lrho,(self.Vdim,self.Vdim))
               
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Relaxation in Hilbert space and Liouville Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

    def SpectralDensity(self,W,tau):
        """
        Spectral Density Function
        Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
        page: 56

        INPUT
        -----
        W: Eigen frequency
        tau: correlation time
        
        OUTPUT
        ------
        return spectral density
        """
        
        return 2*tau/(1 + (W * tau)**2)

    def SpectralDensity_Lb(self,W,tau):
        """
        Spectral Density Function with thermal correction.
        For Lindblad Relaxation
        
        INPUT
        -----
        W: Eigen frequency
        tau: correlation time
        
        OUTPUT
        ------
        return spectral density
        """
        
        #return 2 * tau * np.exp(-0.5 * W * (self.hbar/(self.T * self.kb)))  # W * tau << 1
        return (2 * tau/(1 + W**2 * tau**2)) * np.exp(-0.5 * W * (self.hbar/(self.Lindblad_T * self.kb))) 

    def Spherical_Tensor(self,spin,Rank,m,Sx,Sy,Sz,Sp,Sm):
        """
        Spherical rank tensors
        Reference: Nuclear singlet relaxation by scalar relaxation of the second kind in the slow-fluctuation regime, J. Chem. Phys. 150, 064315 (2019), S.J. Elliot
        
        INPUT
        -----
        spin: List of spin index, example [0, 1] ( 0 corresponds to index of spin 1 and 0 corresponds to index of spin 2 ) or [1,2]
        Rank: rank of spherical tensor
        m: it takes values from -Rank,...,Rank
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Return Value of spherical tensor for corresponding Rank and m value.        
        """
        
        if Rank == 2:

            if m == 0:
                return ((4 * np.matmul(Sz[spin[0]],Sz[spin[1]]) - np.matmul(Sp[spin[0]],Sm[spin[1]]) - np.matmul(Sm[spin[0]],Sp[spin[1]]))/(2 * np.sqrt(6)))  # T(2,0) #
            if m == 1:
                return (-0.5 * (np.matmul(Sz[spin[0]],Sp[spin[1]]) + np.matmul(Sp[spin[0]],Sz[spin[1]]))) # T(2,+1)
            if m == -1:
                return (0.5 * (np.matmul(Sz[spin[0]],Sm[spin[1]]) + np.matmul(Sm[spin[0]],Sz[spin[1]]))) # T(2,-1)
            if m == 2:
                return (0.5 * np.matmul(Sp[spin[0]],Sp[spin[1]])) # T(2,+2)
            if m == -2:
                return (0.5 * np.matmul(Sm[spin[0]],Sm[spin[1]])) # T(2,-2)   
                                
        if Rank == 1:
            if m == 0:
                return Sz[spin[0]]
            if m == 1:
                return (-1/np.sqrt(2)) * Sp[spin[0]]
            if m == -1:
                return (1/np.sqrt(2)) * Sm[spin[0]]                       

    def Spherical_Tensor_Ernst(self,spin,Rank,m,Sx,Sy,Sz,Sp,Sm):
        """
        Spherical rank tensors
        Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
        page: 56
                
        INPUT
        -----
        spin: List of spin index, example [0, 1] ( 0 corresponds to index of spin 1 and 0 corresponds to index of spin 2 ) or [1,2]
        Rank: rank of spherical tensor
        m: it takes values from -Rank,...,Rank
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Return Value of spherical tensor for corresponding Rank and m value.        
        """
        
        if Rank == 2:

            if m == 0:
                return np.sqrt(12/15) * (np.matmul(Sz[spin[0]],Sz[spin[1]]) - 0.25 * np.matmul(Sp[spin[0]],Sm[spin[1]]) - 0.25 *  np.matmul(Sm[spin[0]],Sp[spin[1]]))  # T(2,0) #
            if m == 1:
                return np.sqrt(2/15) * -3.0/2.0 *((np.matmul(Sz[spin[0]],Sp[spin[1]]) + np.matmul(Sp[spin[0]],Sz[spin[1]]))) # T(2,+1)
            if m == -1:
                return np.sqrt(2/15) * -3.0/2.0 * ((np.matmul(Sz[spin[0]],Sm[spin[1]]) + np.matmul(Sm[spin[0]],Sz[spin[1]]))) # T(2,-1)
            if m == 2:
                return np.sqrt(8/15) * -3.0/4.0 * (np.matmul(Sp[spin[0]],Sp[spin[1]])) # T(2,+2)
            if m == -2:
                return np.sqrt(8/15) * -3.0/4.0 * (np.matmul(Sm[spin[0]],Sm[spin[1]])) # T(2,-2) 

    def Spherical_Tensor_Ernst_P(self,spin,Rank,m,Sx,Sy,Sz,Sp,Sm):
        """
        Spherical rank tensors
        Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
        page: 56
                
        INPUT
        -----
        spin: List of spin index, example [0, 1] ( 0 corresponds to index of spin 1 and 0 corresponds to index of spin 2 ) or [1,2]
        Rank: rank of spherical tensor
        m: it takes values from -Rank,...,Rank
        Sx: Spin Operator Sx
        Sy: Spin Operator Sy
        Sz: Spin Operator Sz
        Sp: Spin Operator Sp
        Sm: Spin Operator Sm
        
        OUTPUT
        ------
        Return Value of spherical tensor for corresponding Rank and m value.        
        """
        
        if Rank == 2:

            if m == 10 or m == -10:
                return np.sqrt(12/15) * (np.matmul(Sz[spin[0]],Sz[spin[1]])), 0.0  # T(2,0) # P = 1
            if m == 20 or m == -20:
                return np.sqrt(12/15) * (-0.25 * np.matmul(Sp[spin[0]],Sm[spin[1]])), (self.LarmorF[spin[0]] - self.LarmorF[spin[1]])   # T(2,0) # P = 2
            if m == 30 or m == -30:
                return np.sqrt(12/15) * (-0.25 *  np.matmul(Sm[spin[0]],Sp[spin[1]])), (self.LarmorF[spin[1]] - self.LarmorF[spin[0]])  # T(2,0) # P = 3
                                                
            if m == 11:
                return np.sqrt(2/15) * -3.0/2.0 *((np.matmul(Sz[spin[0]],Sp[spin[1]]))), (self.LarmorF[spin[1]]) # T(2,+1) # P = 1
            if m == 12:
                return np.sqrt(2/15) * -3.0/2.0 *((np.matmul(Sp[spin[0]],Sz[spin[1]]))), (self.LarmorF[spin[0]]) # T(2,+1) # P = 2
                                
            if m == -11:
                return np.sqrt(2/15) * -3.0/2.0 * ((np.matmul(Sz[spin[0]],Sm[spin[1]]))), (-self.LarmorF[spin[1]]) # T(2,-1) # P = 1
            if m == -12:
                return np.sqrt(2/15) * -3.0/2.0 * ((np.matmul(Sm[spin[0]],Sz[spin[1]]))), (-self.LarmorF[spin[0]]) # T(2,-1) # P = 2
                                
            if m == 2:
                return np.sqrt(8/15) * -3.0/4.0 * (np.matmul(Sp[spin[0]],Sp[spin[1]])), (self.LarmorF[spin[0]] + self.LarmorF[spin[1]]) # T(2,+2)
            if m == -2:
                return np.sqrt(8/15) * -3.0/4.0 * (np.matmul(Sm[spin[0]],Sm[spin[1]])), (-self.LarmorF[spin[0]] - self.LarmorF[spin[1]]) # T(2,-2) 

    def EigFreq_ProductOperator_L(self,Hz_L,opBasis_L):
        """
        Compute the eigen frequency of the eigen (operator) basis of the Hamiltonian commutation superoperator (Liouville)
        
        INPUT
        -----
        Hz_L: Commutation superoperator Hamiltonian
        opBasis_L: Eigen Operator
        
        OUTPUT
        ------
        return eigen frequency
        """
        
        #print("Eigen Frequency in Hz")
        return np.trace(self.Adjoint(opBasis_L) @ Hz_L @ opBasis_L).real/(2.0*np.pi)

    def EigFreq_ProductOperator_H(self,Hz,opBasis):
        """
        Compute the eigen frequency of the eigen (operator) basis of the Hamiltonian (Hilbert)
        
        INPUT
        -----
        Hz: Hamiltonian
        opBasis: Eigen Operator
        
        OUTPUT
        ------
        return eigen frequency
        """
        
        #print("Eigen Frequency in Hz")
        return self.InnerProduct(opBasis,self.Commutator(Hz,opBasis)).real/(2.0*np.pi)
        
    def RelaxationRate_H(self,A,B):
        """
        Compute Relaxation rate: <A|RB> / <A|A>
        
        INPUT
        -----
        A: Spin operator
        B: Spin Operator
        
        OUTPUT
        ------
        relaxation rate
        """    

        Rprocess = self.Rprocess

        RelaxOP = self.Relaxation(B)
        
        return self.InnerProduct(A,RelaxOP) / self.InnerProduct(A,A)

    def RelaxationRate_L(self,A,B,Relax_L):
        """
        Compute Relaxation rate: <A|RB> / <A|A>
        
        INPUT
        -----
        A: Spin operator
        B: Spin Operator
        
        OUTPUT
        ------
        relaxation rate
        """    
        
        return (self.Vector_L(A)).T @ Relax_L.real @ self.Vector_L(B) / ((self.Vector_L(A)).T @ self.Vector_L(A))

    def Lindblad_Dissipator(self,A,B):
        """
        Lindbald Dissipator
        
        INPUT
        -----
        A:
        B:
        
        OUTPUT
        ------
        return Lindblad Dissipator
        """

        #return np.kron(A,B.T) - 0.5 * ( np.kron(np.matmul(B,A), np.eye(self.Vdim)) + np.kron(np.eye(self.Vdim), np.matmul(A.T,B.T)) ) 
        return np.kron(A,B.T) - 0.5 * self.AntiCommutationSuperoperator(B @ A)

    def Relaxation(self,rho=None):

        R1 = self.R1
        R2 = self.R2
        R_input = self.R_Matrix
        Rprocess = self.Rprocess

        Sx = self.Sx_
        Sy = self.Sy_ 
        Sz = self.Sz_
        Sp = self.Sp_
        Sm = self.Sm_

        if self.MasterEquation == "Redfield" and self.PropagationSpace == "Hilbert":
            """
            Redfield Relaxation in Hilbert space
            
            ATTENTION
            ---------
            This function is called by Evolution_H(self,rhoeq,rho,Sx,Sy,Sz,Sp,Sm,Hamiltonian,dt,Npoints,method,Rprocess), check it for more informations.
            """
            
            if Rprocess == "No Relaxation":
                """
                No Relaxation
                """
                dim = self.Vdim
                Rso = np.zeros((dim,dim))
                
            if Rprocess == "Phenomenological":
                """
                Phenomenological Relaxation
                """
                dim = self.Vdim 
                Rso = R2 * np.ones((dim,dim))
                np.fill_diagonal(Rso, R1) 
                Rso = 2.0 * np.multiply(Rso,rho)               

            if Rprocess == "Phenomenological Matrix":
                """
                Phenomenological Relaxation
                Relaxation Matrix is given as input
                see function, Relaxation_Phenomenological_Input(R)
                """
                Rso = 2.0 * np.multiply(R_input,rho) 
                
            if Rprocess == "Auto-correlated Random Field Fluctuation":
                """
                Auto-correlated
                Random Field Fluctuation
                """
                omega_R = 1.0e11 # Default: 1.0e11
                dim = self.Vdim
                Rso = np.zeros((dim,dim))
                for i in range(self.Nspins):   
                        
                    #Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.DoubleCommutator(Sz[i],Sz[i],rho) + 0.5 * self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * self.DoubleCommutator(Sp[i],Sm[i],rho) + 0.5 * self.SpectralDensity(-1 * self.LarmorF[i],self.RelaxParDipole_tau) * self.DoubleCommutator(Sm[i],Sp[i],rho)) 
                                
                    Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.DoubleCommutator(Sz[i],self.Adjoint(Sz[i]),rho) + 0.5 * self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * self.DoubleCommutator(Sp[i],self.Adjoint(Sp[i]),rho) + 0.5 * self.SpectralDensity(-1 * self.LarmorF[i],self.RelaxParDipole_tau) * self.DoubleCommutator(Sm[i],self.Adjoint(Sm[i]),rho))

            if Rprocess == "Phenomenological Random Field Fluctuation":
                """
                Auto-correlated
                Random Field Fluctuation
                Phenomenological (R1 and R2 inputs)
                """
                kxy = R1/2.0
                kz = R2 - kxy
                dim = self.Vdim
                Rso = np.zeros((dim,dim))
                for i in range(self.Nspins):
                    Rso = Rso + kz * self.DoubleCommutator(Sz[i],Sz[i],rho) + kxy * self.DoubleCommutator(Sp[i],Sm[i],rho) + kxy * self.DoubleCommutator(Sm[i],Sp[i],rho)
                    
            if Rprocess == "Auto-correlated Dipolar Heteronuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                Rso = np.zeros((self.Vdim,self.Vdim),dtype=np.cdouble)
                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS) 

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [10,10,10,20,20,20,30,30,30,11,11,12,12,-11,-11,-12,-12,2,-2]
                    n = [10,20,30,10,20,30,10,20,30,-11,-12,-11,-12,11,12,11,12,-2,2]
                    for i,l in zip(m,n):
                        StensorRank2, Eigen_Freq = self.Spherical_Tensor_Ernst_P([j,k],2,i,Sx,Sy,Sz,Sp,Sm)
                        StensorRank2_Adjoint, Eigen_Freq_Adjoint = self.Spherical_Tensor_Ernst_P([j,k],2,l,Sx,Sy,Sz,Sp,Sm)
                        Rso = Rso + DDC**2 * self.SpectralDensity(Eigen_Freq,self.RelaxParDipole_tau) * self.DoubleCommutator(StensorRank2,StensorRank2_Adjoint,rho)
                        
                Rso = Rso   

            if Rprocess == "Auto-correlated Dipolar Homonuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                Rso = np.zeros((self.Vdim,self.Vdim),dtype=np.cdouble)
                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS) 

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * self.SpectralDensity(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.DoubleCommutator(self.Spherical_Tensor_Ernst([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor_Ernst([j,k],2,-i,Sx,Sy,Sz,Sp,Sm),rho)   
                        
                Rso = Rso       

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Nuclear singlet relaxation by scalar relaxation of the second kind in the slow-fluctuation regime, J. Chem. Phys. 150, 064315 (2019), S.J. Elliot
                """
                Rso = np.zeros((self.Vdim,self.Vdim),dtype=np.cdouble)
                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * (-1)**i * self.SpectralDensity(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.DoubleCommutator(self.Spherical_Tensor([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([j,k],2,-i,Sx,Sy,Sz,Sp,Sm),rho)
                        
                Rso = Rso * (6/5) * 0.5                
                    
            return 0.5 * Rso 
        
        if self.MasterEquation == "Redfield" and self.PropagationSpace == "Liouville":
            """
            Redfield Relaxation in Liouville Space
            
            INPUT
            -----
            Rprocess: "No Relaxation" or 
                    "Phenomenological" or 
                    "Auto-correlated Random Field Fluctuation" or 
                    "Auto-correlated Dipolar Homonuclear" or 
                    "Cross Correlated CSA - Dipolar Hetronuclear"
                    
            Sx: Spin Operator Sx
            Sy: Spin Operator Sy
            Sz: Spin Operator Sz
            Sp: Spin Operator Sp
            Sm: Spin Operator Sm
            
            OUTPUT
            ------
            Rso: Relaxation Superoperator 
                    
            """      
            
            if Rprocess == "No Relaxation":
                """
                No Relaxation
                """
                Rso = np.zeros((self.Ldim,self.Ldim))
                
            if Rprocess == "Phenomenological":  
                """
                Phenomenological Relaxation
                """
                Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
                np.fill_diagonal(Rso, R)
                
            if Rprocess == "Auto-correlated Random Field Fluctuation":
                """
                Auto-correlated Random Field Fluctuation Relaxation
                """
                omega_R = 1.0e11
                Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
                for i in range(self.Nspins):
                    Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.DoubleCommutationSuperoperator(Sz[i],Sz[i]) + self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * (self.DoubleCommutationSuperoperator(Sp[i],Sm[i]) + self.DoubleCommutationSuperoperator(Sm[i],Sp[i])))

            if Rprocess == "Auto-correlated Dipolar Heteronuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [10,10,10,20,20,20,30,30,30,11,11,12,12,-11,-11,-12,-12,2,-2]
                    n = [10,20,30,10,20,30,10,20,30,-11,-12,-11,-12,11,12,11,12,-2,2]
                    for i,l in zip(m,n):
                        StensorRank2, Eigen_Freq = self.Spherical_Tensor_Ernst_P([j,k],2,i,Sx,Sy,Sz,Sp,Sm)
                        StensorRank2_Adjoint, Eigen_Freq_Adjoint = self.Spherical_Tensor_Ernst_P([j,k],2,l,Sx,Sy,Sz,Sp,Sm)
                        Rso = Rso + DDC**2 * self.SpectralDensity(Eigen_Freq,self.RelaxParDipole_tau) * self.DoubleCommutationSuperoperator(StensorRank2,StensorRank2_Adjoint)
                        
                Rso = Rso   

            if Rprocess == "Auto-correlated Dipolar Homonuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * self.SpectralDensity(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.DoubleCommutationSuperoperator(self.Spherical_Tensor_Ernst([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor_Ernst([j,k],2,-i,Sx,Sy,Sz,Sp,Sm))   
                        
                Rso = Rso       

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Nuclear singlet relaxation by scalar relaxation of the second kind in the slow-fluctuation regime, J. Chem. Phys. 150, 064315 (2019), S.J. Elliot
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * (-1)**i * self.SpectralDensity(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.DoubleCommutationSuperoperator(self.Spherical_Tensor([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([j,k],2,-i,Sx,Sy,Sz,Sp,Sm))
                        
                Rso = Rso * (6/5) * 0.5                
                    
            return 0.5 * Rso

        if self.MasterEquation == "Lindblad" and self.PropagationSpace == "Liouville":
            if Rprocess == "Auto-correlated Dipolar Heteronuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [10,10,10,20,20,20,30,30,30,11,11,12,12,-11,-11,-12,-12,2,-2]
                    n = [10,20,30,10,20,30,10,20,30,-11,-12,-11,-12,11,12,11,12,-2,2]
                    for i,l in zip(m,n):
                        StensorRank2, Eigen_Freq = self.Spherical_Tensor_Ernst_P([j,k],2,i,Sx,Sy,Sz,Sp,Sm)
                        StensorRank2_Adjoint, Eigen_Freq_Adjoint = self.Spherical_Tensor_Ernst_P([j,k],2,l,Sx,Sy,Sz,Sp,Sm)
                        Rso = Rso + DDC**2 * self.SpectralDensity_Lb(Eigen_Freq,self.RelaxParDipole_tau) * self.Lindblad_Dissipator(StensorRank2,StensorRank2_Adjoint)
                        
                Rso = -1 * Rso   

            if Rprocess == "Auto-correlated Dipolar Homonuclear Ernst":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, Richard R Ernst, et.al.
                page: 56             
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * self.SpectralDensity_Lb(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.Lindblad_Dissipator(self.Spherical_Tensor_Ernst([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor_Ernst([j,k],2,-i,Sx,Sy,Sz,Sp,Sm))   
                        
                Rso = -1 * Rso       

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear Auto-correlated
                Dipolar Relaxation
                Extreme Narrowing
                More than 2 spins
                
                Reference: Nuclear singlet relaxation by scalar relaxation of the second kind in the slow-fluctuation regime, J. Chem. Phys. 150, 064315 (2019), S.J. Elliot
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T 
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                for j,k, DDC in zip(Spin_INDEX_1,Spin_INDEX_2,DD_Coupling):
                    m = [-2,-1,0,1,2]
                    for i in m:
                        Rso = Rso + DDC**2 * (-1)**i * self.SpectralDensity_Lb(i * self.LarmorF[0],self.RelaxParDipole_tau) * self.Lindblad_Dissipator(self.Spherical_Tensor([j,k],2,i,Sx,Sy,Sz,Sp,Sm),self.Spherical_Tensor([j,k],2,-i,Sx,Sy,Sz,Sp,Sm))
                        
                Rso = Rso * (-6/5) * 0.5                
                    
            return Rso

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Curve Fitting
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def Exp_Decay(self,x,a,b,c):
        """
        Exponential Decay, function = a exp(-b x) + c
        
        INPUTS
        ------
        x: look function defined
        a: look function defined
        b: look function defined
        c: look function defined
        
        OUTPUT
        ------
        func: return the function defined
        
        """
        return a * np.exp(-b * x) + c

    def Exp_Decay_2(self,x,a,b,c,d,e):
        """
        Exponential Decay, function = a exp(-b x) + c exp(-d x) + e
        
        INPUTS
        ------
        x: look function defined
        a: look function defined
        b: look function defined
        c: look function defined
        d: look function defined
        e: look function defined
        
        OUTPUT
        ------
        func: return the function defined
        
        """
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e

    def Exp_BuildUp(self,x,a,b,c):
        """
        Exponential Build-up, function = c - (c - a) * exp(-b x)
        
        INPUTS
        ------
        x: look function defined
        a: look function defined
        b: look function defined
        c: look function defined
        
        OUTPUT
        ------
        func: return the function defined
        
        """
        return c - (c - a) * np.exp(-b * x)
        
    def Fitting_LeastSquare(self,func, xdata, ydata):    
        """
        Non-linear least squares to fit a function
        
        INPUTS
        ------
        func: Function used to fit
        xdata: X data
        ydata: Y data
        
        OUTPUT
        ------
        popt: Optimal values for the parameters so that the sum of the squared residuals of func(xdata, *popt) - ydata is minimized
        pcov:The estimated approximate covariance of popt.
        
        Note
        ----
        For more information read: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        """
        popt, pcov = curve_fit(func, xdata, ydata)
        
        return popt, pcov
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Frequency Analyzer
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
class Fanalyzer:
    def __init__(self,Mx,My,ax,fig,line1,line2,line3,line4,vline1,vline2,vline3,vline4,text1,text2):
        """
        Multi-mode Maser Analyzer
        Time domain (Figure 1,1) to Frequency domain (Figure 1,2)
        Frequency domain (Figure 2,1) to Time domain (Figure 2,2)
        
        Select region of FID (Figure 1,1), its Fourier Transform (in red) will be shown in Figure 1,2. (Blue for entire Fourier Transform)
        Select region of frequencies (Figure 2,1), corresponding time signal (in red) will be shown in Figure 2,2. (Blue for total FID) 
        """
        
        self.x1, self.y1 = line1.get_data()
        self.x2, self.y2 = line2.get_data()
        self.x3, self.y3 = line3.get_data()
        self.x4, self.y4 = line4.get_data()
        self.dt = self.x1[1] - self.x1[0]
        self.fs = 1.0/self.dt
        self.ax = ax
        self.fig = fig
        self.vline1 = vline1
        self.vline2 = vline2
        self.text1 = text1
        self.vline3 = vline3
        self.vline4 = vline4 
        self.text2 = text2
        self.Mx = Mx
        self.My = My
        self.Mt = Mx + 1j * My

    def button_press(self,event):
        if event.inaxes is self.ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1in
            x1in = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	    
        if event.inaxes is self.ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3in
            x3in = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)

        if event.inaxes is self.ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2in
            x2in = x2
            self.vline1.set_xdata([x2in])
            plt.draw()

        if event.inaxes is self.ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4in
            x4in = x4
            self.vline3.set_xdata([x4in])
            plt.draw()

    def button_release(self,event):
        if event.inaxes is self.ax[0,0]:
            x1, y1 = event.xdata, event.ydata
            global x1fi
            x1fi = min(np.searchsorted(self.x1, x1), len(self.x1) - 1)
	        
            spectrum = np.fft.fft(self.Mt[x1in:x1fi])
            spectrum = np.fft.fftshift(spectrum)
            freq = np.linspace(-self.fs/2,self.fs/2,spectrum.shape[-1])
            la = self.ax[0,1].get_lines()
            la[-1].remove()
            line2, = self.ax[0,1].plot(self.x2,np.absolute(self.y2),"-", color='blue')
            line, = self.ax[0,1].plot(freq,spectrum,"-", color='red')
            #line, = self.ax[0,1].plot(freq,np.absolute(spectrum),"-", color='red')
            plt.draw()

        if event.inaxes is self.ax[1,0]:
            x3, y3 = event.xdata, event.ydata
            global x3fi
            x3fi = min(np.searchsorted(self.x3, x3), len(self.x3) - 1)
            y3 = self.y3
            print(y3.shape)
            window = np.zeros((y3.shape[-1]))
            window[x3in:x3fi] = 1.0
            sig = np.fft.ifftshift(y3*window)
            sig = np.fft.ifft(sig)
            t = np.linspace(0,self.dt*y3.shape[-1],y3.shape[-1])
            lb = self.ax[1,1].get_lines()
            lb[-1].remove()
            line4, = self.ax[1,1].plot(self.x4,self.y4,'-', color='blue')
            line, = self.ax[1,1].plot(t,sig,"-", color='red')
            plt.draw()

        if event.inaxes is self.ax[0,1]:
            x2, y2 = event.xdata, event.ydata
            global x2fi
            x2fi = x2
            self.vline2.set_xdata([x2fi])
            self.text1.set_text(f'Freq={abs(x2fi-x2in):1.5f} Hz')
            plt.draw()

        if event.inaxes is self.ax[1,1]:
            x4, y4 = event.xdata, event.ydata
            global x4fi
            x4fi = x4
            self.vline4.set_xdata([x4fi])
            self.text2.set_text(f'Time={abs(x4fi-x4in):1.5f} s')
            plt.draw()        
                         
