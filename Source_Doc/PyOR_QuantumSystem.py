"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module defines the `QuantumSystem` class, which represents composite 
    quantum systems consisting of multiple interacting particles or subsystems.

    The `QuantumSystem` class provides tools for managing the total system Hilbert space, 
    constructing composite states and operators, and simulating evolution under 
    system-wide Hamiltonians.
"""


# -------------- Package Imports --------------
try:
    from .PyOR_QuantumObject import QunObj
    from . import PyOR_PhysicalConstants
    from . import PyOR_SpinQuantumNumber
    from . import PyOR_Gamma
    from . import PyOR_QuadrupoleMoment
    from . import PyOR_Particle
except ImportError:
    from PyOR_QuantumObject import QunObj
    import PyOR_PhysicalConstants
    import PyOR_SpinQuantumNumber
    import PyOR_Gamma
    import PyOR_QuadrupoleMoment
    import PyOR_Particle


import numpy as np
from numpy import linalg as lina
import sympy as sp
from sympy.physics.quantum.cg import CG

import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.linalg as la

import os, sys, time, re, math
from fractions import Fraction
from functools import reduce
from collections import defaultdict

import numba
from numba import njit, cfunc

from IPython.display import display, Latex, Math

# -------------- Main Class --------------

class QuantumSystem:
    """
    QuantumSystem: Central class in PyOR for defining and manipulating multi-spin quantum systems.

    This class sets up the quantum system based on a list of spins, initializes physical and
    numerical parameters, manages spin operators, relaxation mechanisms, acquisition settings,
    plotting preferences, and state evolution.

    Parameters
    ----------
    SpinList : dict
        Dictionary mapping spin labels to spin types (e.g., {'I': '1H', 'S': '13C'})
    PrintDefault : bool, optional
        If True, prints all default system settings and parameters.

    Attributes
    ----------
    SpinList : dict
        User-defined spin system.
    SpinDic : list
        List of spin keys from the dictionary.
    SpinName : np.array
        List of spin names (e.g., ['1H', '13C']).
    SpinIndex : dict
        Dictionary mapping spin label to index.
    slist : np.array
        List of spin quantum numbers.
    Sdim : np.array
        Dimensions of each individual spin Hilbert space.
    Vdim : int
        Dimension of the total Hilbert space.
    Ldim : int
        Dimension of the Liouville space (Vdim^2).
    hbarEQ1 : bool
        Toggle to treat ℏ = 1 in Hamiltonians.
    MatrixTolarence : float
        Elements smaller than this are considered zero in matrices.
    Gamma : list
        Gyromagnetic ratios of spins.
    B0 : float
        Static magnetic field (Tesla).
    OFFSET, OMEGA_RF : dict
        Offset and rotating frame frequencies.
    Jlist : np.ndarray
        J-coupling matrix.
    Dipole_Pairs : list
        List of dipolar-coupled spin index pairs.
    Various attributes...
        Many more attributes initialized for relaxation, plotting, acquisition, etc.
    """

    def __init__(self, SpinList, PrintDefault=True):
        # Spin system definition
        self.SpinList = SpinList
        self.SpinDic = list(SpinList.keys())
        self.SpinIndex = {value: index for index, value in enumerate(self.SpinDic)}
        self.SpinName = np.array(list(SpinList.values()))

        # Extract spin quantum numbers from names (e.g., '1H' -> 1/2)
        SPINLIST = [PyOR_SpinQuantumNumber.spin(name) for name in self.SpinName]
        self.slist = np.array(SPINLIST)
        self.Nspins = len(self.slist)

        # Hilbert space dimensions
        self.Sdim = np.array([np.arange(-s, s + 1, 1).shape[-1] for s in self.slist])
        self.Vdim = np.prod(self.Sdim)
        self.Ldim = self.Vdim ** 2
        self.Inverse2PI = 1.0 / (2.0 * np.pi)

        # Proton Larmor Frequency
        self.L100 = 2.349495 # T        
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

        # Default configuration messages
        if PrintDefault:
            print("\nPyOR default parameters/settings")
            print("--------------------------------")

        # ----------------- Constants & Defaults -----------------
        self.hbarEQ1 = True
        self.MatrixTolarence = 1.0e-10
        self.Gamma = [PyOR_Gamma.gamma(name) for name in self.SpinName]
        self.B0 = None
        self.OMEGA_RF = {key: 0 for key in self.SpinList}
        self.OFFSET = {key: 0 for key in self.SpinList}
        self.LARMOR_F = {key: 0 for key in self.SpinList}
        self.print_Larmor = True
        self.Jlist = np.zeros((self.Nspins, self.Nspins))
        self.Dipole_Pairs = []
        self.Dipole_DipolarAlpabet = []
        self.DipoleAngle = []
        self.DipolebIS = []

        # ----------------- Basis of the Spin Operators -----------------
        self.Basis_SpinOperators = "Zeeman"
        self.Basis_SpinOperators_TransformationMatrix = None # Unitary transformation matrix should be QunObj

        # ----------------- Temperature -----------------
        self.I_spintemp = {key: 0 for key in self.SpinList}
        self.F_spintemp = {key: 0 for key in self.SpinList}

        # ----------------- Propagation -----------------
        self.PropagationSpace = "Hilbert"

        # ----------------- Acquisition -----------------
        self.AcqDT = 0.0001
        self.AcqFS = 1.0 / self.AcqDT
        self.AcqAQ = 5.0

        # ----------------- Relaxation -----------------
        self.MasterEquation = "Redfield"
        self.Rprocess = "No Relaxation"
        self.R1 = 0.0
        self.R2 = 0.0
        if self.PropagationSpace == "Hilbert":
            self.R_Matrix = np.zeros((self.Vdim, self.Vdim), dtype=np.double)
        else:
            self.R_Matrix = np.zeros((self.Ldim, self.Ldim), dtype=np.double)
        self.RelaxParDipole_tau = 0.0
        self.RelaxParDipole_bIS = []
        self.Lindblad_Temp = 300
        self.InverseSpinTemp = False
        self.Maser_TempGradient = False
        self.Lindblad_TempGradient = 0.0
        self.Lindblad_InitialInverseTemp = 0.0
        self.Lindblad_FinalInverseTemp = 0.0

        # ----------------- ODE Solver -----------------
        self.PropagationMethod = "ODE Solver"
        self.OdeMethod = 'RK45'
        self.ODE_atol = 1.0e-10
        self.ODE_rtol = 1.0e-10

        # ----------------- Radiation Damping -----------------
        self.Rdamping = False
        self.RD_xi = {key: 0 for key in self.SpinList}
        self.RD_phase = {key: 0 for key in self.SpinList}

        # ----------------- Noise -----------------
        self.NGaussian = False
        self.N_mean = 0.0
        self.N_std = 1.0e-8
        self.N_length = 1

        # ----------------- Plotting -----------------
        self.PlotFigureSize = (5, 5)
        self.PlotFontSize = 5
        self.PlotXlimt = (None, None)
        self.PlotYlimt = (None, None)
        self.PlotArrowlength = 0.5
        self.PlotLinwidth = 2

        # ----------------- Misc -----------------
        self.Shift_para = 0.0
        self.Dipole_Shift = False
        self.SparseM = False
        self.ShapeFunc = None
        self.ShapeParOmega = None
        self.ShapeParPhase = None
        self.ShapeParFreq = None
        self.RowColOrder = 'C'
        self.DTYPE_C = np.csingle
        self.DTYPE_F = np.single
        self.ORDER_MEMORY = "C"

        # ------ Optional Printouts ------
        if PrintDefault:
            self._PrintSystemDefaults()

    def _PrintSystemDefaults(self):
        """
        Internal helper method to print default settings for the initialized system.
        """
        print("\nDefine energy units: hbarEQ1 =", self.hbarEQ1)
        print("\nDefine the matrix tolerance (threshold for zeroing small elements): MatrixTolarence =", self.MatrixTolarence)
        print("\nDefine the gyromagnetic ratios: Gamma =", self.Gamma)
        print("\nDefine the static field along Z: B0 =", self.B0)
        print("\nDefine rotating frame frequency: OMEGA_RF =", self.OMEGA_RF)
        print("\nDefine the offset frequencies of the spins: OFFSET =", self.OFFSET)
        print("\nDo you want to print the Larmor frequency: print_Larmor =", self.print_Larmor)
        print("\nDefine the J coupling: Jlist =\n", self.Jlist)
        print("\nDefine the spin pairs dipolar coupled: Dipole_Pairs =", self.Dipole_Pairs)
        print("\nDefine the spin pairs Zeeman truncation: Dipole_DipolarAlpabet =", self.Dipole_DipolarAlpabet)
        print("\nDefine the spin pairs dipole angle (theta, phi): DipoleAngle =", self.DipoleAngle)
        print("\nDefine the spin pairs coupling constants: DipolebIS =", self.DipolebIS)
        
        print("\nSpin temperatures")
        print("----------------")
        print("Initial spin temperature of individual spins: I_spintemp =", self.I_spintemp)
        print("Final spin temperature of individual spins: F_spintemp =", self.F_spintemp)

        print("\nDefine propagation space <<Hilbert>> or <<Liouville>>: PropagationSpace =", self.PropagationSpace)

        print("\nAcquisition parameters")
        print("----------------------")
        print("Dwell time: AcqDT =", self.AcqDT)
        print("Sampling frequency: AcqFS =", self.AcqFS)
        print("Acquisition time: AcqAQ =", self.AcqAQ)

        print("\nRelaxation process")
        print("------------------")
        print("Master equation <<Redfield>> or <<Lindblad>>: MasterEquation =", self.MasterEquation)
        print("Define relaxation process: Rprocess =", self.Rprocess)
        print("Longitudinal relaxation rate R1 =", self.R1)
        print("Transverse relaxation rate R2 =", self.R2)
        print("Relaxation matrix (phenomenological): R_Matrix =\n", self.R_Matrix)
        print("Dipolar relaxation tau_c: RelaxParDipole_tau =", self.RelaxParDipole_tau)
        print("Dipolar relaxation couplings: RelaxParDipole_bIS =", self.RelaxParDipole_bIS)
        print("Lindblad master equation temperature: Lindblad_Temp =", self.Lindblad_Temp)

        print("\nLindblad special cases")
        print("----------------------")
        print("Inverse spin temperature: InverseSpinTemp =", self.InverseSpinTemp)
        print("Maser temperature gradient active: Maser_TempGradient =", self.Maser_TempGradient)
        print("Temperature gradient dT/dt: Lindblad_TempGradient =", self.Lindblad_TempGradient)
        print("Initial inverse temp: Lindblad_InitialInverseTemp =", self.Lindblad_InitialInverseTemp)
        print("Final inverse temp: Lindblad_FinalInverseTemp =", self.Lindblad_FinalInverseTemp)

        print("\nPropagation")
        print("-----------")
        print("Propagation method: PropagationMethod =", self.PropagationMethod)

        print("\nODE Solver")
        print("-----------")
        print("Method: OdeMethod =", self.OdeMethod)
        print("Absolute tolerance: ODE_atol =", self.ODE_atol)
        print("Relative tolerance: ODE_rtol =", self.ODE_rtol)

        print("\nRadiation Damping")
        print("-----------------")
        print("Enable damping: Rdamping =", self.Rdamping)
        print("Gain values: RD_xi =", self.RD_xi)
        print("Phase values: RD_phase =", self.RD_phase)

        print("\nGaussian Noise")
        print("-----------------")
        print("Enable noise: NGaussian =", self.NGaussian)
        print("Mean: N_mean =", self.N_mean)
        print("Standard deviation: N_std =", self.N_std)

        print("\nPlotting")
        print("--------")
        print("Figure size: PlotFigureSize =", self.PlotFigureSize)
        print("Font size: PlotFontSize =", self.PlotFontSize)
        print("X-axis limit: PlotXlimt =", self.PlotXlimt)
        print("Y-axis limit: PlotYlimt =", self.PlotYlimt)
        print("Arrow length (Bloch): PlotArrowlength =", self.PlotArrowlength)
        print("Line width: PlotLinwidth =", self.PlotLinwidth)

        print("\nSparse Matrix usage:", self.SparseM)

        print("\nShape Pulse")
        print("-----------")
        print("Pulse function: ShapeFunc =", self.ShapeFunc)
        print("Amplitude: ShapeParOmega =", self.ShapeParOmega)
        print("Phase: ShapeParPhase =", self.ShapeParPhase)
        print("Frequency: ShapeParFreq =", self.ShapeParFreq)

        print("\nDensity Matrix Vectorization: RowColOrder =", self.RowColOrder)
        print("\nData Types: Complex =", self.DTYPE_C, ", Float =", self.DTYPE_F, ", Order =", self.ORDER_MEMORY)

    # ---------------------------------------------------------

    def PyOR_Version(self):
        """
        Print version info for PyOR and its dependencies.
        """
        print("PyOR Python On Resonance")
        print("\nVersion: Jeener")
        print("\nMotto: Everybody can simulate NMR")
        print("\nAuthor: Vineeth Francis Thalakottoor")
        print("\nEmail: vineethfrancis.physics@gmail.com")
        print("\nPackage Versions")
        print("----------------")
        print("Numpy version:", np.__version__)
        print("Scipy version:", scipy.__version__)
        print("Sympy version:", sp.__version__)
        print("Numba version:", numba.__version__)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Unit Conversions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def PPMtoHz(self, ppm, ref):
        """
        Convert chemical shift from PPM to Hz.

        Parameters
        ----------
        ppm : float
            Chemical shift in ppm.
        ref : float
            Reference frequency in Hz.

        Returns
        -------
        float
            Frequency in Hz.
        """
        ref = ref / (2 * np.pi)
        return ppm * ref * 1.0e-6

    def HztoPPM(self, freq_sample, ref):
        """
        Convert frequency from Hz to PPM.

        Parameters
        ----------
        freq_sample : float
            Frequency in Hz.
        ref : float
            Reference frequency in Hz.

        Returns
        -------
        float
            Chemical shift in ppm.
        """
        ref = ref / (2 * np.pi)
        return freq_sample / ref * 1.0e6

    def Convert_EnergyTOFreqUnits(self, H):
        """
        Convert Hamiltonian from energy units (Joules) to angular frequency units (rad/s).

        Parameters
        ----------
        H : float or ndarray
            Hamiltonian in energy units.

        Returns
        -------
        float or ndarray
            Hamiltonian in angular frequency units.
        """
        hbar = PyOR_PhysicalConstants.constants("hbar")
        return H / hbar

    def Convert_FreqUnitsTOEnergy(self, H):
        """
        Convert Hamiltonian from angular frequency units to energy units (Joules).

        Parameters
        ----------
        H : float or ndarray
            Hamiltonian in angular frequency units.

        Returns
        -------
        float or ndarray
            Hamiltonian in energy units.
        """
        hbar = PyOR_PhysicalConstants.constants("hbar")
        return H * hbar

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Initialize spin operators and particles
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Initialize(self):
        """
        Initialize the quantum system:
        - Compute spin operators (Sx, Sy, Sz, etc.)
        - Populate particle-related properties (gamma, quadrupole, etc.)
        """
        self.SpinOperator(PrintDefault=False)
        self.ParticleParameters()

    def Update(self):
        """
        Update system settings after parameter changes.

        Useful after modifying B0, OMEGA_RF, OFFSET, etc.
        Recomputes internal attributes used in simulation and spin labels.
        """
        for i in self.SpinDic:
            self.OMEGA_RF[i] = -1 * self.Gamma[self.SpinIndex[i]] * self.B0

        self.OmegaRF = [self.OMEGA_RF[key] for key in self.SpinList]
        self.Offset = [self.OFFSET[key] for key in self.SpinList]
        self.Ispintemp = [self.I_spintemp[key] for key in self.SpinList]
        self.Fspintemp = [self.F_spintemp[key] for key in self.SpinList]
        self.RDxi = [self.RD_xi[key] for key in self.SpinList]
        self.RDphase = [self.RD_phase[key] for key in self.SpinList]

        # Map Dipole_Pairs from label -> index
        self.DipolePairs = []
        for spin_pair in self.Dipole_Pairs:
            index_pair = (self.SpinIndex[spin_pair[0]], self.SpinIndex[spin_pair[1]])
            self.DipolePairs.append(index_pair)

        self.AcqFS = 1.0 / self.AcqDT

        print("Rotating frame frequencies:", self.OMEGA_RF)
        print("Offset frequencies:", self.OFFSET)
        print("Initial spin temperatures:", self.I_spintemp)
        print("Final spin temperatures:", self.F_spintemp)
        print("Radiation damping gain:", self.RD_xi)
        print("Radiation damping phase:", self.RD_phase)
        print(f"\nRprocess = {self.Rprocess}")
        print(f"RelaxParDipole_tau = {self.RelaxParDipole_tau}")
        print(f"DipolePairs = {self.Dipole_Pairs}")
        print(f"RelaxParDipole_bIS = {self.RelaxParDipole_bIS}")

        self.IndividualThermalDensityMatrix()

    def JcoupleValue(self, x, y, value):
        """
        Set scalar J coupling constant between two spins.

        Parameters
        ----------
        x : str
            Label of spin 1.
        y : str
            Label of spin 2.
        value : float
            J coupling constant in Hz.
        """
        self.Jlist[self.SpinIndex[x]][self.SpinIndex[y]] = value

    def ParticleParameters(self):
        """
        Initialize particle properties for each spin.

        Automatically retrieves constants from PyOR_Particle.
        """
        for sdic, sname in zip(self.SpinDic, self.SpinName):
            setattr(self, f"{sdic}", PyOR_Particle.particle(sname))

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Spin Operators
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def SpinOperatorsSingleSpin(self, X):
        """
        Generate spin operators (Sx, Sy, Sz) for a single spin quantum number.

        Parameters
        ----------
        X : float
            Spin quantum number (e.g., 1/2, 1, 3/2).

        Returns
        -------
        np.ndarray
            A 3D array with shape (3, dim, dim) representing [Sx, Sy, Sz] matrices.

        Reference:
        ----------
        Quantum Mechanics: Concepts and Applications, Nouredine Zettili.
        """
        hbar = PyOR_PhysicalConstants.constants("hbar")

        ms = np.arange(X, -X - 1, -1)  # Magnetic quantum numbers: [X, X-1, ..., -X]
        dim = ms.shape[-1]

        # Initialize matrices
        SingleSpin = np.zeros((3, dim, dim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sp = np.zeros((dim, dim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sn = np.zeros((dim, dim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)

        # Identity and shifted identity matrices
        Id = np.eye(dim)
        Idp = np.triu(np.roll(Id, 1, axis=1), k=1)  # Delta(m', m+1)
        Idn = np.tril(np.roll(Id, -1, axis=1), k=1)  # Delta(m', m-1)

        # Populate S+, S-, Sz
        for i in range(dim):
            for j in range(dim):
                SingleSpin[2][i][j] = hbar * ms[j] * Id[i][j]
                Sp[i][j] = np.sqrt(X * (X + 1) - ms[j] * (ms[j] + 1)) * Idp[i][j]
                Sn[i][j] = np.sqrt(X * (X + 1) - ms[j] * (ms[j] - 1)) * Idn[i][j]

        # Sx = (S+ + S-) / 2
        # Sy = -i(S+ - S-) / 2
        SingleSpin[0] = hbar * 0.5 * (Sp + Sn)
        SingleSpin[1] = hbar * (-1j / 2) * (Sp - Sn)

        # Set hbar = 1 if specified
        if self.hbarEQ1:
            SingleSpin = SingleSpin / hbar

        return SingleSpin

    def SpinOperator(self, PrintDefault=False):
        """
        Generate spin operators Sx, Sy, Sz, Sp, Sm for all spins in the system.

        Also assigns:
        - `self.Sx_`, `self.Sy_`, `self.Sz_`, `self.Sp_`, `self.Sm_` : ndarray
        - self.Jsquared (total angular momentum operator): QunObj
        - Individual spin operators as class attributes (e.g., Ix, Iy, Iz, etc.)
        """
        Sx = np.zeros((self.Nspins, self.Vdim, self.Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sy = np.zeros((self.Nspins, self.Vdim, self.Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sz = np.zeros((self.Nspins, self.Vdim, self.Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sp = np.zeros((self.Nspins, self.Vdim, self.Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sm = np.zeros((self.Nspins, self.Vdim, self.Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)

        # Compute spin operators via Kronecker product
        for i in range(self.Nspins):
            VSlist_x, VSlist_y, VSlist_z = [], [], []
            for j in range(self.Nspins):
                VSlist_x.append(np.eye(self.Sdim[j]))
                VSlist_y.append(np.eye(self.Sdim[j]))
                VSlist_z.append(np.eye(self.Sdim[j]))

            S_single = self.SpinOperatorsSingleSpin(self.slist[i])
            VSlist_x[i], VSlist_y[i], VSlist_z[i] = S_single[0], S_single[1], S_single[2]

            Sx[i] = reduce(np.kron, VSlist_x)
            Sy[i] = reduce(np.kron, VSlist_y)
            Sz[i] = reduce(np.kron, VSlist_z)

            Sp[i] = Sx[i] + 1j * Sy[i]
            Sm[i] = Sx[i] - 1j * Sy[i]

        if self.Basis_SpinOperators == "Hamiltonian eigen states":
            Sx = self.BasisChange_SpinOperators_Local(Sx,self.Basis_SpinOperators_TransformationMatrix)
            Sy = self.BasisChange_SpinOperators_Local(Sy,self.Basis_SpinOperators_TransformationMatrix)
            Sz = self.BasisChange_SpinOperators_Local(Sz,self.Basis_SpinOperators_TransformationMatrix)
            Sp = self.BasisChange_SpinOperators_Local(Sp,self.Basis_SpinOperators_TransformationMatrix)
            Sm = self.BasisChange_SpinOperators_Local(Sm,self.Basis_SpinOperators_TransformationMatrix)

        # Save operators
        self.Sx_ = Sx
        self.Sy_ = Sy
        self.Sz_ = Sz
        self.Sp_ = Sp
        self.Sm_ = Sm

        Jsq = (
            np.matmul(np.sum(Sx, axis=0), np.sum(Sx, axis=0)) +
            np.matmul(np.sum(Sy, axis=0), np.sum(Sy, axis=0)) +
            np.matmul(np.sum(Sz, axis=0), np.sum(Sz, axis=0))
        )

        if self.Basis_SpinOperators == "Hamiltonian eigen states":
            Jsq = self.BasisChange_Operator_Local(Jsq,self.Basis_SpinOperators_TransformationMatrix)

        self.Jsq_ = Jsq
        setattr(self, "Jsq", QunObj(Jsq, PrintDefault=PrintDefault))

        # Assign individual spin operators as class attributes (e.g., Ix, Iy, Iz...)
        for idx, spin in enumerate(self.SpinDic):
            setattr(self, f"{spin}x", QunObj(Sx[idx], PrintDefault=PrintDefault))
            setattr(self, f"{spin}y", QunObj(Sy[idx], PrintDefault=PrintDefault))
            setattr(self, f"{spin}z", QunObj(Sz[idx], PrintDefault=PrintDefault))
            setattr(self, f"{spin}p", QunObj(Sp[idx], PrintDefault=PrintDefault))
            setattr(self, f"{spin}m", QunObj(Sm[idx], PrintDefault=PrintDefault))
            setattr(self, f"{spin}id", QunObj(np.eye(self.Vdim), PrintDefault=PrintDefault))

        self.SpinOperator_Sub(PrintDefault=PrintDefault)

    def SpinOperator_Sub(self, PrintDefault=False):
        """
        Generate subsystem spin operators for each individual spin.
        
        Stores operators with suffix `_sub` for each spin label.
        Example: Ix_sub, Iy_sub, Iz_sub, Ip_sub, Im_sub, Iid_sub
        """
        for idx, spin in enumerate(self.SpinDic):
            Sx, Sy, Sz = self.SpinOperatorsSingleSpin(self.slist[idx])
            Sp = Sx + 1j * Sy
            Sm = Sx - 1j * Sy

            setattr(self, f"{spin}x_sub", QunObj(Sx, PrintDefault=PrintDefault))
            setattr(self, f"{spin}y_sub", QunObj(Sy, PrintDefault=PrintDefault))
            setattr(self, f"{spin}z_sub", QunObj(Sz, PrintDefault=PrintDefault))
            setattr(self, f"{spin}p_sub", QunObj(Sp, PrintDefault=PrintDefault))
            setattr(self, f"{spin}m_sub", QunObj(Sm, PrintDefault=PrintDefault))
            setattr(self, f"{spin}id_sub", QunObj(np.eye(Sx.shape[0]), PrintDefault=PrintDefault))

    def SpinOperator_SpinQunatulNumber_List(self, SpinQNlist):
        """
        Generate full-system spin operators from a list of spin quantum numbers.

        Parameters
        ----------
        SpinQNlist : list of float
            List of spin quantum numbers [s1, s2, ..., sn].

        Returns
        -------
        tuple of np.ndarray
            Arrays of shape (n, dim, dim) for Sx, Sy, and Sz.
        """
        S = np.asarray(SpinQNlist, dtype=float)
        Nspins = S.shape[-1]
        Sdim = np.array([np.arange(-s, s + 1).shape[-1] for s in S])
        Vdim = np.prod(Sdim)

        Sx = np.zeros((Nspins, Vdim, Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sy = np.zeros((Nspins, Vdim, Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)
        Sz = np.zeros((Nspins, Vdim, Vdim), dtype=self.DTYPE_C, order=self.ORDER_MEMORY)

        for i in range(Nspins):
            VSlist_x = [np.eye(Sdim[j]) for j in range(Nspins)]
            VSlist_y = [np.eye(Sdim[j]) for j in range(Nspins)]
            VSlist_z = [np.eye(Sdim[j]) for j in range(Nspins)]

            SingleSpin = self.SpinOperatorsSingleSpin(S[i])
            VSlist_x[i], VSlist_y[i], VSlist_z[i] = SingleSpin[0], SingleSpin[1], SingleSpin[2]

            Sx[i] = reduce(np.kron, VSlist_x)
            Sy[i] = reduce(np.kron, VSlist_y)
            Sz[i] = reduce(np.kron, VSlist_z)

        return Sx, Sy, Sz

    def MagQnuSingle(self, X):
        """
        Return magnetic quantum number values for an individual spin.

        Parameters
        ----------
        X : float
            Spin quantum number.

        Returns
        -------
        np.ndarray
            Array of magnetic quantum numbers: [X, X-1, ..., -X].
        """
        return np.arange(X, -X - 1, -1)

    def MagQnuSystem(self):
        """
        Return total magnetic quantum numbers (Sz expectation) for each Zeeman state.

        Returns
        -------
        np.ndarray
            Diagonal values of total Sz.
        """
        Sz = self.Sz_
        return (np.sum(Sz, axis=0).real).diagonal()

    def StateZeeman(self, MagQunList):
        """
        Construct a Zeeman state vector from magnetic quantum numbers of individual spins.

        Parameters
        ----------
        MagQunList : dict
            Dictionary of spin labels to their magnetic quantum numbers.

        Returns
        -------
        QunObj
            Tensor product state corresponding to the given Zeeman state.
        """
        SpinDic = list(MagQunList.keys())
        magQun = np.array(list(MagQunList.values()))

        eigenvectors = []

        for i in range(len(SpinDic)):
            spin_label = SpinDic[i]
            attribute_name = f"{spin_label}z_sub"

            eigval, eigvec = self.Eigen_Split(getattr(self, attribute_name))
            index = np.where(np.isclose(eigval, magQun[i], atol=1e-8))[0]

            if index.size > 0:
                eigenvectors.append(eigvec[index[0]])

        if len(eigenvectors) == 1:
            return eigenvectors[0]
        else:
            EV = eigenvectors[0]
            for i in range(1, len(eigenvectors)):
                EV = EV.TensorProduct(eigenvectors[i])
            return EV

    def States(self, DicList):
        """
        Construct tensor product state(s) from a list of Zeeman or coupled spin dictionaries.

        Supports both simple Zeeman basis and Clebsch-Gordan combined basis (under testing).

        Parameters
        ----------
        DicList : list
            List of spin label dictionaries or nested dictionaries with CG specification.

        Returns
        -------
        QunObj
            Tensor product of eigenstates.
        """
        eigenvectors = []
        eigenvectors_multi = []

        for d in DicList:
            if any(isinstance(value, dict) for value in d.values()):
                new_dict = d["New"]
                old_dict = d["Old"]
                l_value = new_dict["l"]
                m_value = new_dict["m"]
                select_value = new_dict["Select_l"]
                SpinDic = list(old_dict.keys())
                New_SpinList = [PyOR_SpinQuantumNumber.spin(self.SpinList[i]) for i in old_dict]

                Sx, Sy, Sz = self.SpinOperator_SpinQunatulNumber_List(New_SpinList)
                S_ = np.matmul(np.sum(Sx, axis=0), np.sum(Sx, axis=0)) + \
                     np.matmul(np.sum(Sy, axis=0), np.sum(Sy, axis=0)) + \
                     np.matmul(np.sum(Sz, axis=0), np.sum(Sz, axis=0))

                QunObj_S = QunObj(S_)
                QunObj_Sz = QunObj(np.sum(Sz, axis=0))
                _, eigenvector_objs = self.Eigen_Split(QunObj_S + QunObj_Sz)

                for vec in eigenvector_objs:
                    ll, mm = self.State_SpinQuantumNumber_SpinOperators(vec, QunObj_S, QunObj_Sz)
                    if l_value == ll and m_value == mm:
                        eigenvectors_multi.append(vec)

                eigenvectors.append(eigenvectors_multi[select_value])
                eigenvectors_multi = []
            else:
                eigenvectors.append(self.StateZeeman(d))

        EV = eigenvectors[0]
        for i in range(1, len(eigenvectors)):
            EV = EV.TensorProduct(eigenvectors[i])
        return EV

    def Bracket(self, X: 'QunObj', A: 'QunObj', Y: 'QunObj') -> float:
        """
        Compute the bracket ⟨X|A|Y⟩ / ⟨X|Y⟩.

        Used for angular momentum evaluation.

        Parameters
        ----------
        X, A, Y : QunObj
            Operators and state vectors.

        Returns
        -------
        float
            Result of ⟨X|A|Y⟩ normalized by ⟨X|Y⟩.
        """
        if not isinstance(X, QunObj) or not isinstance(A, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("All arguments must be QunObj instances.")

        num = np.matmul(X.data.conj().T, np.matmul(A.data, Y.data))
        denom = np.matmul(X.data.conj().T, Y.data)
        return (num / denom)[0].real

    def State_SpinQuantumNumber_SpinOperators(self, A: 'QunObj', Ssq: 'QunObj', Sz: 'QunObj'):
        """
        Extract spin quantum numbers from a given state and spin operators.

        Parameters
        ----------
        A : QunObj
            State vector.
        Ssq : QunObj
            Total spin-squared operator.
        Sz : QunObj
            Total Sz operator.

        Returns
        -------
        tuple
            (Total spin l, magnetic quantum number m)
        """
        XX = self.Bracket(A, Ssq, A)
        magQunNumber = round(self.Bracket(A, Sz, A)[0] * 2) / 2

        a, b, c = 1, 1, -np.round(XX, decimals=10)
        disc = b**2 - 4*a*c

        if disc >= 0:
            roots = [(-b + math.sqrt(disc)) / (2*a), (-b - math.sqrt(disc)) / (2*a)]
            real_pos = [r for r in roots if r >= 0]
            if real_pos:
                return round(real_pos[0] * 2) / 2, magQunNumber
        return None

    def State_SpinQuantumNumber(self, A: 'QunObj'):
        """
        Find total spin and magnetic quantum number for a system state.

        Parameters
        ----------
        A : QunObj
            State vector.

        Returns
        -------
        tuple
            (Total spin l, magnetic quantum number m)
        """
        X = self.Bracket(A, self.Jsq, A)
        magQunNumber = round(self.Bracket(A, QunObj(np.sum(self.Sz_, axis=0)), A)[0] * 2) / 2

        a, b, c = 1, 1, -X
        disc = b**2 - 4*a*c

        if disc >= 0:
            roots = [(-b + math.sqrt(disc)) / (2*a), (-b - math.sqrt(disc)) / (2*a)]
            real_pos = [r for r in roots if r >= 0]
            if real_pos:
                l = round(real_pos[0] * 2) / 2
                print(f"Spin quantum number = {l}, magnetic quantum number = {magQunNumber}")
                return l, magQunNumber
        return None

    def Eigen(self, A: QunObj) -> tuple:
        """
        Compute eigenvalues and eigenvectors of a quantum object.

        Parameters
        ----------
        A : QunObj
            The quantum object (operator) to decompose.

        Returns
        -------
        tuple
            (eigenvalues as ndarray, eigenvectors as QunObj)
        """
        if not isinstance(A, QunObj):
            raise TypeError("Input must be a QunObj instance.")

        eigenvalues, eigenvectors = la.eig(A.data)
        return eigenvalues, QunObj(eigenvectors)

    def Eigen_Split(self, A: QunObj) -> tuple:
        """
        Compute eigenvalues and return eigenvectors as a list of QunObj.

        Parameters
        ----------
        A : QunObj
            Quantum operator.

        Returns
        -------
        tuple
            (eigenvalues as ndarray, eigenvectors as list of QunObj column vectors)
        """
        if not isinstance(A, QunObj):
            raise TypeError("Input must be a QunObj instance.")

        eigenvalues, eigenvectors = la.eig(A.data)
        eigenvector_objs = [QunObj(vec.reshape(-1, 1)) for vec in eigenvectors.T]
        return eigenvalues, eigenvector_objs

    def ZeemanBasis_Ket(self):
        """
        Return list of basis kets as strings for the full system.

        Returns
        -------
        list of str
            Zeeman basis state labels like `|1/2,-1/2⟩`.
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()[f"Spin_List_{i}"] = []

        for idx, j in enumerate(self.slist):
            for k in self.MagQnuSingle(j):
                locals()[f"Spin_List_{idx}"].append(f"|{Fraction(j)},{Fraction(k)}⟩")

        def Combine(A, B):
            return [l + m for l in A for m in B]

        if self.Nspins == 1:
            return locals()["Spin_List_0"]
        for n in range(self.Nspins - 1):
            locals()[f"Spin_List_{n+1}"] = Combine(locals()[f"Spin_List_{n}"], locals()[f"Spin_List_{n+1}"])
        return locals()[f"Spin_List_{self.Nspins - 1}"]

    def ZeemanBasis_Bra(self):
        """
        Return list of basis bras as strings for the full system.

        Returns
        -------
        list of str
            Zeeman basis state labels like ⟨1/2,-1/2|.
        """
        LABEL = []
        LABEL_temp = []
        for i in range(self.Nspins):
            locals()[f"Spin_List_{i}"] = []

        for idx, j in enumerate(self.slist):
            for k in self.MagQnuSingle(j):
                locals()[f"Spin_List_{idx}"].append(f"⟨{Fraction(j)},{Fraction(k)}|")

        def Combine(A, B):
            return [l + m for l in A for m in B]

        if self.Nspins == 1:
            return locals()["Spin_List_0"]
        for n in range(self.Nspins - 1):
            locals()[f"Spin_List_{n+1}"] = Combine(locals()[f"Spin_List_{n}"], locals()[f"Spin_List_{n+1}"])
        return locals()[f"Spin_List_{self.Nspins - 1}"]

    def IndividualThermalDensityMatrix(self):
        """
        Initialize and assign the individual spin thermal density matrices.

        For each spin, constructs the density matrix using Boltzmann distribution
        in the Zeeman basis and stores it as an attribute.
        """
        hbar = 1.054e-34
        kb = 1.380e-23

        for sdic in self.SpinDic:
            gamma = self.Gamma[self.SpinIndex[sdic]]
            offset = self.Offset[self.SpinIndex[sdic]]
            temp = self.Ispintemp[self.SpinIndex[sdic]]
            Sz_sub = getattr(self, f"{sdic}z_sub", None).data

            W = -gamma * self.B0 - 2 * np.pi * offset
            H = hbar * W * Sz_sub / (kb * temp)

            rho = expm(-H)
            rho_normalized = rho / np.trace(rho)
            setattr(self, f"{sdic}rho", QunObj(rho_normalized))

    def BasisChange_Operator_Local(self, O, U):
        """
        Transform an operator using the given transformation matrix.

        Parameters
        ----------
        O : np.ndarray
            Operator in the original basis.
        U : QunObj
            Transformation matrix as a QunObj.

        Returns
        -------
        np.ndarray
            Operator in the new basis.
        """
        if not isinstance(O, np.ndarray):
            raise TypeError("O must be a NumPy array.")
        if not isinstance(U, QunObj):
            raise TypeError("U must be an instance of QunObj.")

        U_dag = self.Adjoint(U.data)
        return U_dag @ O @ U.data 

    def BasisChange_SpinOperators_Local(self, Sop, U):
        """
        Transform an array of spin operators using a transformation matrix.

        Parameters
        ----------
        Sop : np.ndarray
            Array of shape (N, dim, dim) containing spin operators (e.g., [Sx, Sy, Sz]).
        U : QunObj
            Transformation matrix as a QunObj.

        Returns
        -------
        np.ndarray
            Transformed spin operators as an array of shape (N, dim, dim).
        """
        if not isinstance(Sop, np.ndarray):
            raise TypeError("Sop must be a NumPy array.")

        if Sop.ndim != 3:
            raise ValueError("Sop must be a 3D NumPy array of shape (N, dim, dim).")

        if not isinstance(U, QunObj):
            raise TypeError("U must be an instance of QunObj.")

        U_data = U.data
        U_dag = self.Adjoint(U_data)
        Sop_N = np.empty_like(Sop, dtype=self.DTYPE_C)

        for i in range(Sop.shape[0]):
            Sop_N[i] = U_dag @ Sop[i] @ U_data

        return Sop_N
    
    def Adjoint(self, A):
        """
        Compute the adjoint (Hermitian conjugate) of an operator.

        Parameters
        ----------
        A : ndarray
            Operator or state vector.

        Returns
        -------
        ndarray
            Hermitian conjugate of the input.
        """
        return A.T.conj()    
