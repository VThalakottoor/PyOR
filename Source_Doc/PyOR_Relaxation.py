"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module defines the `Relaxation` class, which provides methods to model 
    relaxation processes in magnetic resonance simulations.

    The `Relaxation` class includes functionalities for simulating longitudinal (T1) 
    and transverse (T2) relaxation, relaxation superoperators, and decoherence 
    mechanisms relevant to spin dynamics.
"""

from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
    HAVE_THREADPOOLCTL = True
except Exception:
    HAVE_THREADPOOLCTL = False


import numpy as np
from numpy import linalg as lina
from scipy.interpolate import interp1d

import re
from io import StringIO
from scipy import sparse

try:
    from . import PyOR_PhysicalConstants
    from . import PyOR_Rotation
    from .PyOR_QuantumObject import QunObj
    from .PyOR_Hamiltonian import Hamiltonian
    from .PyOR_Commutators import Commutators
    from .PyOR_Basis import Basis
except ImportError:
    import PyOR_PhysicalConstants
    import PyOR_Rotation
    from PyOR_QuantumObject import QunObj
    from PyOR_Hamiltonian import Hamiltonian
    from PyOR_Commutators import Commutators
    from PyOR_Basis import Basis    


class RelaxationProcess:
    def __init__(self, class_QS):
        self.class_QS = class_QS
        self.class_COMM = Commutators()
        self.hbar = PyOR_PhysicalConstants.constants("hbar")
        self.mu0 = PyOR_PhysicalConstants.constants("mu0")
        self.kb = PyOR_PhysicalConstants.constants("kb")
        self.class_Ham = Hamiltonian(self.class_QS)
        self.LarmorF = self.class_Ham.LarmorF

        # Dimensions
        self.Vdim = class_QS.Vdim
        self.Ldim = class_QS.Ldim

        # System Setup
        self.MasterEquation = self.class_QS.MasterEquation
        self.PropagationSpace = self.class_QS.PropagationSpace
        self.Rprocess = self.class_QS.Rprocess
        self.DipolePairs = class_QS.DipolePairs
        self.RelaxParDipole_bIS = class_QS.RelaxParDipole_bIS
        self.RelaxParDipole_tau = class_QS.RelaxParDipole_tau
        self.R1 = class_QS.R1
        self.R2 = class_QS.R2
        self.R_Matrix = class_QS.R_Matrix  
        self.Nspins = class_QS.Nspins 
        self.SparseM = class_QS.SparseM

    def Adjoint(self, A):
        """
        Return adjoint (Hermitian conjugate) of operator A
        """
        return A.T.conj()

    def InnerProduct(self, A, B):
        """
        Inner product of two operators or vectors: ⟨A|B⟩
        """
        return np.trace(np.matmul(A.T.conj(), B))

    def Vector_L(self, X):
        """
        Vectorize the operator X for Liouville space calculations.
        """
        dim = self.class_QS.Vdim
        return np.reshape(X, (dim**2, -1))

    def SpectralDensity(self, W, tau):
        """
        Spectral density function J(ω) for Redfield theory.

        Reference:
        Richard R. Ernst et al., "Principles of Nuclear Magnetic Resonance in One and Two Dimensions", p.56

        Parameters:
        -----------
        W : float
            Angular frequency
        tau : float
            Correlation time (s)

        Returns:
        --------
        float
            Spectral density J(ω)
        """
        return 2 * tau / (1 + (W * tau)**2)

    def SpectralDensity_Lb(self, W, tau):
        """
        Spectral density with thermal correction for Lindblad master equation.

        Reference:
        C. Bengs, M.H. Levitt, JMR 310 (2020), Eq. 140

        Parameters:
        -----------
        W : float
            Angular frequency
        tau : float
            Correlation time (s)

        Returns:
        --------
        float
            Thermally corrected spectral density J(ω)
        """
        return (2 * tau / (1 + (W * tau)**2)) * np.exp(-0.5 * W * self.hbar / (self.class_QS.Lindblad_Temp * self.kb))
            

    def Spherical_Tensor(self, spin, Rank, m, Sx, Sy, Sz, Sp, Sm):
        """
        Spherical tensor components (Rank 1 and 2).
        
        Reference:
        S.J. Elliott, J. Chem. Phys. 150, 064315 (2019)

        Parameters:
        -----------
        spin : list of int
            Spin indices (e.g. [0,1])
        Rank : int
            Tensor rank (1 or 2)
        m : int
            Tensor component (-Rank to +Rank)

        Returns:
        --------
        np.ndarray
            Tensor operator T(Rank, m)
        """
        Rprocess = self.Rprocess
        if Rprocess in ("Auto-correlated Dipolar Homonuclear", "Auto-correlated Dipolar Homonuclear Parallel"):        
            if Rank == 2:
                if m == 0:
                    return (4 * Sz[spin[0]] @ Sz[spin[1]] - Sp[spin[0]] @ Sm[spin[1]] - Sm[spin[0]] @ Sp[spin[1]]) / (2 * np.sqrt(6))
                if m == 1:
                    return -0.5 * (Sz[spin[0]] @ Sp[spin[1]] + Sp[spin[0]] @ Sz[spin[1]])
                if m == -1:
                    return 0.5 * (Sz[spin[0]] @ Sm[spin[1]] + Sm[spin[0]] @ Sz[spin[1]])
                if m == 2:
                    return 0.5 * (Sp[spin[0]] @ Sp[spin[1]])
                if m == -2:
                    return 0.5 * (Sm[spin[0]] @ Sm[spin[1]])

    def Spherical_Tensor_P(self, spin, Rank, m, Sx, Sy, Sz, Sp, Sm):
        """
        Spherical tensor components (Rank 1 and 2).
        
        Reference:
        S.J. Elliott, J. Chem. Phys. 150, 064315 (2019)

        Parameters:
        -----------
        spin : list of int
            Spin indices (e.g. [0,1])
        Rank : int
            Tensor rank (1 or 2)
        m : int
            Tensor component (-Rank to +Rank)

        Returns:
        --------
        np.ndarray
            Tensor operator T(Rank, m)
        """
        Rprocess = self.Rprocess
        if Rprocess in ("Auto-correlated Dipolar Heteronuclear", "Auto-correlated Dipolar Heteronuclear Parallel"):
            if Rank == 2:
                if m == 10 or m==-10:
                    return (4 * Sz[spin[0]] @ Sz[spin[1]]) / (2 * np.sqrt(6)), 0.0
                if m == 20 or m==-20:
                    return (-1 * Sp[spin[0]] @ Sm[spin[1]]) / (2 * np.sqrt(6)), self.LarmorF[spin[0]] - self.LarmorF[spin[1]]
                if m == 30 or m==-30:
                    return (-1 * Sm[spin[0]] @ Sp[spin[1]]) / (2 * np.sqrt(6)), self.LarmorF[spin[1]] - self.LarmorF[spin[0]]
                            
                if m == 11:
                    return -0.5 * (Sz[spin[0]] @ Sp[spin[1]]), self.LarmorF[spin[1]]
                if m == 12:
                    return -0.5 * (Sp[spin[0]] @ Sz[spin[1]]), self.LarmorF[spin[0]]
                            
                if m == -11:
                    return 0.5 * (Sz[spin[0]] @ Sm[spin[1]]), -self.LarmorF[spin[1]]
                if m == -12:
                    return 0.5 * (Sm[spin[0]] @ Sz[spin[1]]), -self.LarmorF[spin[0]]            

                if m == 2:
                    return 0.5 * (Sp[spin[0]] @ Sp[spin[1]]), self.LarmorF[spin[0]] + self.LarmorF[spin[1]]
                if m == -2:
                    return 0.5 * (Sm[spin[0]] @ Sm[spin[1]]), -self.LarmorF[spin[0]] - self.LarmorF[spin[1]]

        if Rprocess == "Auto-correlated Quadrupole":
            if Rank == 2:
                if m == 0:
                    return (3.0 * Sz[spin[0]] @ Sz[spin[0]] - 2) / (np.sqrt(6)), 0.0
                            
                if m == 1:
                    return -0.5 * (Sz[spin[0]] @ Sp[spin[0]] + Sp[spin[0]] @ Sz[spin[0]]), self.LarmorF[spin[0]]
                if m == -1:
                    return 0.5 * (Sz[spin[0]] @ Sm[spin[0]] + Sm[spin[0]] @ Sz[spin[0]]), self.LarmorF[spin[0]]            

                if m == 2:
                    return 0.5 * (Sp[spin[0]] @ Sp[spin[0]]), 2.0 * self.LarmorF[spin[0]]
                if m == -2:
                    return 0.5 * (Sm[spin[0]] @ Sm[spin[0]]), -2.0 * self.LarmorF[spin[0]]
                
        if Rprocess == "Auto-correlated CSA":
            if Rank == 1:
                if m == 0:
                    return (2.0/np.sqrt(6)) * Sz[spin[0]], 0.0
                if m == 1:
                    return (-1.0 / 2.0) * Sp[spin[0]], self.LarmorF[spin[0]]
                if m == -1:
                    return (1.0 / 2.0) * Sm[spin[0]], -self.LarmorF[spin[0]]

    def EigFreq_ProductOperator_L(self, Hz_L, opBasis_L):
        """
        Compute the eigenfrequency of a product operator in Liouville space.

        Parameters:
        -----------
        Hz_L : np.ndarray
            Liouvillian Hamiltonian superoperator
        opBasis_L : np.ndarray
            Vectorized operator (in Liouville space)

        Returns:
        --------
        float
            Eigenfrequency in Hz
        """
        return np.trace(self.Adjoint(opBasis_L) @ Hz_L @ opBasis_L).real / (2.0 * np.pi)

    def EigFreq_ProductOperator_H(self, Hz, opBasis):
        """
        Compute eigenfrequency in Hilbert space.

        Parameters:
        -----------
        Hz : np.ndarray
            Hamiltonian (Hilbert space)
        opBasis : np.ndarray
            Operator to analyze

        Returns:
        --------
        float
            Eigenfrequency in Hz
        """
        return self.InnerProduct(opBasis, self.class_COMM.Commutator(Hz, opBasis)).real / (2.0 * np.pi)

    def RelaxationRate_H(self, AQ, BQ):
        """
        Compute relaxation rate in Hilbert space:
        <A|R(B)> / <A|A>

        Parameters:
        -----------
        AQ : QunObj
            First operator A
        BQ : QunObj
            Second operator B

        Returns:
        --------
        float
            Relaxation rate (unit depends on RProcess model)
        """
        A = AQ.data
        B = BQ.data

        RelaxOP = self.Relaxation(B)
        return self.InnerProduct(A, RelaxOP) / self.InnerProduct(A, A)

    def RelaxationRate_L(self, AQ, BQ, Relax_LQ):
        """
        Compute relaxation rate in Liouville space:
        <A|R|B> / <A|A>

        Parameters:
        -----------
        AQ : QunObj
            Operator A (Hilbert space)
        BQ : QunObj
            Operator B (Hilbert space)
        Relax_LQ : QunObj
            Relaxation superoperator in Liouville space

        Returns:
        --------
        float
            Relaxation rate
        """
        A = AQ.data
        B = BQ.data
        Relax_L = Relax_LQ.data

        AVec = self.Vector_L(A)
        BVec = self.Vector_L(B)

        return AVec.T @ Relax_L.real @ BVec / (AVec.T @ AVec)

    def Lindblad_Dissipator(self, A, B):
        """
        Compute the Lindblad dissipator in Liouville space.

        Parameters:
        -----------
        A : np.ndarray
            Operator A
        B : np.ndarray
            Operator B

        Returns:
        --------
        np.ndarray
            Lindblad dissipator superoperator
        """
        return np.kron(A, B.T) - 0.5 * self.class_COMM.AntiCommutationSuperoperator(B @ A)

    def Lindblad_Dissipator_Hilbert(self, A, B, rho):
        """
        Compute Lindblad dissipator directly in Hilbert space.

        Parameters:
        -----------
        A : np.ndarray
            Operator A
        B : np.ndarray
            Operator B
        rho : np.ndarray
            Density matrix

        Returns:
        --------
        np.ndarray
            Result of applying Lindblad dissipator
        """
        return A @ rho @ B - 0.5 * self.class_COMM.AntiCommutator(B @ A, rho)
    
    def Relaxation_CoherenceDecay(self, coherence_orders, relaxa_rate, diagonal_relaxa_rate=0, default_rate=0):
        """
        Apply relaxation decay to selected coherence orders, with proper separation of true diagonal elements.

        Parameters:
        - coherence_orders (list or set): coherence orders to apply relaxation to (off-diagonal).
        - relaxa_rate (float): relaxation rate to apply to specified off-diagonal coherence orders.
        - diagonal_relaxa_rate (float): relaxation rate for true diagonal elements (population terms).
        - default_rate (float): relaxation rate for all other coherence orders.

        Returns:
        - Modified coherence Zeeman array with relaxation rates applied.
        """
        BS = Basis(self.class_QS)
        Basis_Zeeman, dic_Zeeman, coh_Zeeman, coh_Zeeman_arrayQ = BS.ProductOperators_Zeeman()
        coh_Zeeman_array = coh_Zeeman_arrayQ.data

        # Get matrix shape and create indices (matrix is assumed square)
        n = coh_Zeeman_array.shape[0]
        rows, cols = np.indices((n, n))

        # Masks
        mask_true_diagonal = (rows == cols)
        mask_relax = np.isin(coh_Zeeman_array, coherence_orders) & (~mask_true_diagonal)
        mask_default = ~(mask_true_diagonal | mask_relax)

        # Apply rates unconditionally
        coh_Zeeman_array[mask_true_diagonal] = diagonal_relaxa_rate
        coh_Zeeman_array[mask_relax] = relaxa_rate
        coh_Zeeman_array[mask_default] = default_rate

        return QunObj(coh_Zeeman_array)


    def Relaxation(self,rho=None,Rprocess = None):
        """
        Compute the relaxation superoperator or apply relaxation to the given density matrix
        based on the selected relaxation process, propagation space, and master equation.

        Parameters
        ----------
        rho : ndarray, optional
            The input density matrix to apply relaxation on. Required for Hilbert space propagation.

        Rprocess : str, optional
            The relaxation model to apply. If None, the default self.Rprocess is used.
            Supported options include:
            - "No Relaxation"
            - "Phenomenological"
            - "Phenomenological Matrix"
            - "Auto-correlated Random Field Fluctuation"
            - "Phenomenological Random Field Fluctuation"
            - "Auto-correlated Dipolar Heteronuclear Ernst"
            - "Auto-correlated Dipolar Homonuclear Ernst"
            - "Auto-correlated Dipolar Homonuclear"

        Returns
        -------
        ndarray or QunObj
            The relaxation superoperator (in Liouville space), or its action on the density matrix
            (in Hilbert space), depending on the system settings.

        Notes
        -----
        - MasterEquation and PropagationSpace determine how the relaxation is calculated.
        - This function supports Redfield and Lindblad equations.
        - It handles both Hilbert and Liouville space propagation.
        - For "Hilbert" space, this returns the applied relaxation: R(rho)
        - For "Liouville" space, this returns the relaxation superoperator R
        - Reference: Principles of Nuclear Magnetic Resonance in One and Two Dimensions, R. R. Ernst et al.
        """

        if Rprocess == None:
            Rprocess = self.Rprocess
        self.Rprocess = Rprocess

        R1 = self.R1
        R2 = self.R2
        R_input = self.class_QS.R_Matrix.data 

        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        Sz = self.class_QS.Sz_
        Sp = self.class_QS.Sp_
        Sm = self.class_QS.Sm_

        # ==================================================
        # Redfield Equation - Hilbert Space
        # ==================================================
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
                return 0.5 * Rso
                            
            if Rprocess == "Phenomenological":
                """
                Phenomenological Relaxation
                """
                dim = self.Vdim 
                Rso = R2 * np.ones((dim,dim))
                np.fill_diagonal(Rso, R1) 
                Rso = 2.0 * np.multiply(Rso,rho)               
                return 0.5 * Rso
            
            if Rprocess == "Phenomenological Matrix":
                """
                Phenomenological Relaxation
                Relaxation Matrix is given as input
                see function, Relaxation_Phenomenological_Input(R)
                """
                Rso = 2.0 * np.multiply(R_input,rho) 
                return 0.5 * Rso
                            
            if Rprocess == "Auto-correlated Random Field Fluctuation":
                """
                Auto-correlated
                Random Field Fluctuation
                """
                omega_R = 1.0e11 # Default: 1.0e11
                dim = self.Vdim
                Rso = np.zeros((dim,dim))
                for i in range(self.Nspins):   
                        
                    #Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.class_COMMf.DoubleCommutator(Sz[i],Sz[i],rho) + 0.5 * self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutator(Sp[i],Sm[i],rho) + 0.5 * self.SpectralDensity(-1 * self.LarmorF[i],self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutator(Sm[i],Sp[i],rho)) 
                                
                    Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutator(Sz[i],self.Adjoint(Sz[i]),rho) + 0.5 * self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutator(Sp[i],self.Adjoint(Sp[i]),rho) + 0.5 * self.SpectralDensity(-1 * self.LarmorF[i],self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutator(Sm[i],self.Adjoint(Sm[i]),rho))
                return 0.5 * Rso
            
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
                    Rso = Rso + kz * self.class_COMM.DoubleCommutator(Sz[i],Sz[i],rho) + kxy * self.class_COMM.DoubleCommutator(Sp[i],Sm[i],rho) + kxy * self.class_COMM.DoubleCommutator(Sm[i],Sp[i],rho)                    
                return 0.5 * Rso
            
            if Rprocess == "Auto-correlated Dipolar Heteronuclear":
                """
                Heteronuclear auto-correlated dipolar relaxation (Redfield, Hilbert space)

                Uses split tensors from Spherical_Tensor_P and ONLY auto-correlated partner pairs.
                Includes adjoint phase consistent with T_{2m}† = (-1)^m T_{2,-m}.
                """
                Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                # ONLY auto-correlated partner pairs (no cross terms)
                pairs = [
                    (10, 10),              # m=0 (SzSz)
                    (20, 30), (30, 20),     # m=0 (SpSm <-> SmSp)
                    (11, -11), (-11, 11),   # m=±1 (SzSp <-> SzSm) on spin 2
                    (12, -12), (-12, 12),   # m=±1 (SpSz <-> SmSz) on spin 1
                    (2, -2), (-2, 2),       # m=±2 (SpSp <-> SmSm)
                ]

                # Phase for adjoint relation:
                # |m|=1 parts -> -1 ; m=0 and |m|=2 -> +1
                def phase(label):
                    return -1.0 if label in (11, -11, 12, -12) else 1.0

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for a_lbl, b_lbl in pairs:
                        A, wA = self.Spherical_Tensor_P([j, k], 2, a_lbl, Sx, Sy, Sz, Sp, Sm)
                        B, wB = self.Spherical_Tensor_P([j, k], 2, b_lbl, Sx, Sy, Sz, Sp, Sm)

                        Jw = self.SpectralDensity(wA, self.RelaxParDipole_tau)

                        Rso += (DDC**2) * phase(a_lbl) * Jw * self.class_COMM.DoubleCommutator(A, B, rho)

                Rso = Rso * (6.0/5.0) * 0.5
                return 0.5 * Rso

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear auto-correlated dipolar relaxation (Redfield, Hilbert space)

                Conventions:
                - self.LarmorF[...] is angular frequency ω (rad/s)
                - pairing uses T_{2m} and T_{2,-m}
                - adjoint handled via phase factor: T_{2m}† = (-1)^m T_{2,-m}
                """
                Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmn = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        w = m * self.LarmorF[0]  # keep your existing global ω0 choice
                        Jw = self.SpectralDensity(w, self.RelaxParDipole_tau)

                        Rso += (DDC**2) * ((-1)**m) * Jw * self.class_COMM.DoubleCommutator(Tm, Tmn, rho)

                Rso = Rso * (6.0/5.0) * 0.5
                return 0.5 * Rso

            if Rprocess == "Auto-correlated Dipolar Heteronuclear Parallel":
                """
                Same model as 'Auto-correlated Dipolar Heteronuclear' but parallelized over DipolePairs using joblib.
                Uses per-pair local accumulation and reduces at the end.
                """

                Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                pairs = [
                    (10, 10),              # m=0 (SzSz)
                    (20, 30), (30, 20),     # m=0 (SpSm <-> SmSp)
                    (11, -11), (-11, 11),   # m=±1 (SzSp <-> SzSm) on spin 2
                    (12, -12), (-12, 12),   # m=±1 (SpSz <-> SmSz) on spin 1
                    (2, -2), (-2, 2),       # m=±2 (SpSp <-> SmSm)
                ]

                def Phase(label):
                    return -1.0 if label in (11, -11, 12, -12) else 1.0

                def OnePairContribution(j, k, DDC):
                    Rloc = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for a_lbl, b_lbl in pairs:
                        A, wA = self.Spherical_Tensor_P([j, k], 2, a_lbl, Sx, Sy, Sz, Sp, Sm)
                        B, _  = self.Spherical_Tensor_P([j, k], 2, b_lbl, Sx, Sy, Sz, Sp, Sm)
                        Jw = self.SpectralDensity(wA, self.RelaxParDipole_tau)
                        Rloc += (DDC**2) * Phase(a_lbl) * Jw * self.class_COMM.DoubleCommutator(A, B, rho)
                    return Rloc

                if self.ParallelEnabled():
                    with self.Blas1ThreadCtx():
                        parts = Parallel(n_jobs=-1, prefer="processes")(
                            delayed(OnePairContribution)(j, k, DDC)
                            for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling)
                        )
                    Rso = np.sum(parts, axis=0)
                else:
                    for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                        Rso += OnePairContribution(j, k, DDC)

                Rso = Rso * (6.0/5.0) * 0.5
                return 0.5 * Rso

            if Rprocess == "Auto-correlated Dipolar Homonuclear Parallel":
                """
                Homonuclear auto-correlated dipolar relaxation (Redfield, Hilbert space),
                parallelized over DipolePairs using joblib.

                Conventions kept identical to the non-parallel version:
                - self.LarmorF[0] used as global ω0 (rad/s)
                - DD_Coupling = 2π * RelaxParDipole_bIS
                - pairing uses T_{2m} and T_{2,-m}
                - adjoint handled via (-1)^m factor
                """

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]
                w0 = self.LarmorF[0]

                def OnePairContribution(j, k, DDC):
                    Rloc = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmn = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        w = m * w0
                        Jw = self.SpectralDensity(w, self.RelaxParDipole_tau)

                        Rloc += (DDC**2) * ((-1)**m) * Jw * self.class_COMM.DoubleCommutator(Tm, Tmn, rho)
                    return Rloc

                if self.ParallelEnabled():
                    with self.Blas1ThreadCtx():
                        parts = Parallel(n_jobs=-1, prefer="processes")(
                            delayed(OnePairContribution)(j, k, DDC)
                            for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling)
                        )
                    Rso = np.sum(parts, axis=0)
                else:
                    Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                        Rso += OnePairContribution(j, k, DDC)

                Rso = Rso * (6.0/5.0) * 0.5
                return 0.5 * Rso             

        # ==================================================
        # Redfield Equation - Liouville Space
        # =================================================        
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
                np.fill_diagonal(Rso, R1)
                
            if Rprocess == "Auto-correlated Random Field Fluctuation":
                """
                Auto-correlated Random Field Fluctuation Relaxation
                """
                omega_R = 1.0e11
                Rso = np.zeros((self.Ldim,self.Ldim),dtype=np.cdouble)
                for i in range(self.Nspins):
                    Rso = Rso + omega_R * (self.SpectralDensity(0,self.RelaxParDipole_tau) * self.class_COMM.DoubleCommutationSuperoperator(Sz[i],Sz[i]) + self.SpectralDensity(self.LarmorF[i],self.RelaxParDipole_tau) * (self.class_COMM.DoubleCommutationSuperoperator(Sp[i],Sm[i]) + self.class_COMM.DoubleCommutationSuperoperator(Sm[i],Sp[i])))                  

            if Rprocess == "Auto-correlated Dipolar Heteronuclear":
                """
                Heteronuclear auto-correlated dipolar relaxation (Redfield, Liouville space)

                Uses split tensors from Spherical_Tensor_P and ONLY auto-correlated partner pairs.
                Includes adjoint phase consistent with T_{2m}† = (-1)^m T_{2,-m}.
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                # ONLY auto-correlated partner pairs (no cross terms)
                pairs = [
                    (10, 10),              # m=0
                    (20, 30), (30, 20),     # m=0 flip-flop parts
                    (11, -11), (-11, 11),   # m=±1 (spin 2 SQ)
                    (12, -12), (-12, 12),   # m=±1 (spin 1 SQ)
                    (2, -2), (-2, 2),       # m=±2 (DQ)
                ]

                # Phase implementing adjoint relation for each label family
                # |m|=1 components -> -1 ; m=0 and |m|=2 -> +1
                def phase(label):
                    return -1.0 if label in (11, -11, 12, -12) else 1.0

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for a_lbl, b_lbl in pairs:
                        A, wA = self.Spherical_Tensor_P([j, k], 2, a_lbl, Sx, Sy, Sz, Sp, Sm)
                        B, wB = self.Spherical_Tensor_P([j, k], 2, b_lbl, Sx, Sy, Sz, Sp, Sm)

                        Jw = self.SpectralDensity(wA, self.RelaxParDipole_tau)

                        Rso = Rso + (DDC**2) * phase(a_lbl) * Jw * self.class_COMM.DoubleCommutationSuperoperator(A, B)

                Rso = Rso * (6.0/5.0) * 0.5

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear auto-correlated dipolar relaxation (Redfield, Liouville space)

                Conventions:
                - self.LarmorF[...] is angular frequency ω (rad/s)
                - pairing uses T_{2m} and T_{2,-m}
                - adjoint handled via phase factor: T_{2m}† = (-1)^m T_{2,-m}
                """
                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmn = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        w = m * self.LarmorF[0]  # keep your existing global ω0 choice
                        Jw = self.SpectralDensity(w, self.RelaxParDipole_tau)

                        Rso = Rso + (DDC**2) * ((-1)**m) * Jw * self.class_COMM.DoubleCommutationSuperoperator(Tm, Tmn)

                Rso = Rso * (6.0/5.0) * 0.5

            if self.SparseM:
                return QunObj(Rso * 0.5)
            else:
                return QunObj(0.5 * Rso)

        # ==================================================
        # Lindblad Equation - Liouville Space
        # ==================================================
        if self.MasterEquation == "Lindblad" and self.PropagationSpace == "Liouville":

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
                np.fill_diagonal(Rso, R1)      

            if Rprocess == "Auto-correlated Dipolar Heteronuclear":
                """
                Heteronuclear auto-correlated dipolar relaxation (Lindblad, Liouville space)

                - Uses Spherical_Tensor_P split components with their partner components.
                - Uses thermally corrected spectral density SpectralDensity_Lb.
                - Converts Hz -> rad/s for spectral density.
                - Includes spherical-tensor adjoint phase consistent with T_{2m}† = (-1)^m T_{2,-m}.
                """

                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T

                # Keep your existing convention (bIS in Hz -> convert to rad/s):
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                # Partner pairs (auto-correlated)
                pairs = [
                    (10, 10),              # m = 0 part
                    (20, 30), (30, 20),     # m = 0 flip-flop parts
                    (11, -11), (-11, 11),   # m = ±1 parts
                    (12, -12), (-12, 12),   # m = ±1 parts
                    (2, -2), (-2, 2),       # m = ±2 parts
                ]

                # Phase factor implementing the adjoint relation
                # rank-2: (-1)^m with m in {0, ±1, ±2}
                def phase(label):
                    if label in (2, -2):           # m = ±2
                        return 1.0
                    if label in (11, -11, 12, -12):  # m = ±1
                        return -1.0
                    # 10,20,30 are m=0 components
                    return 1.0

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for i, l in pairs:
                        A, fA = self.Spherical_Tensor_P([j, k], 2, i, Sx, Sy, Sz, Sp, Sm)
                        B, fB = self.Spherical_Tensor_P([j, k], 2, l, Sx, Sy, Sz, Sp, Sm)

                        Rso = Rso + (DDC**2) * phase(i) * self.SpectralDensity_Lb(fA, self.RelaxParDipole_tau) * self.Lindblad_Dissipator(A, B)

                Rso = Rso * (-6.0/5.0) * 0.5                            

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear Auto-correlated Dipolar Relaxation (Lindblad, Liouville space)

                This implementation is intentionally written to MATCH the existing
                'Auto-correlated Dipolar Homonuclear test' block exactly:
                - uses LarmorF[0] for all pairs
                - uses DD_Coupling = 2π * RelaxParDipole_bIS
                - includes (-1)**m factor
                - uses Spherical_Tensor([j,k],2,m) with partner m->-m
                - final scaling (-6/5)*0.5
                """

                if self.SparseM:
                    Rso = sparse.csc_matrix(np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble))
                else:
                    Rso = np.zeros((self.Ldim, self.Ldim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T

                # Keep the SAME coupling convention as the test:
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]

                # Keep the SAME frequency convention as the test:
                w0 = self.LarmorF[0]   # (you said LarmorF is in rad/s)

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmd = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        Jw = self.SpectralDensity_Lb(m * w0, self.RelaxParDipole_tau)

                        Rso = Rso + (DDC**2) * ((-1)**m) * Jw * self.Lindblad_Dissipator(Tm, Tmd)

                Rso = Rso * (-6.0/5.0) * 0.5

            return QunObj(Rso)   

        # ==================================================
        # Lindblad Equation - Hilbert Space
        # ==================================================
        if self.MasterEquation == "Lindblad" and self.PropagationSpace == "Hilbert":

            if Rprocess == "No Relaxation":
                """
                No Relaxation
                """
                dim = self.Vdim
                Rso = np.zeros((dim,dim))
                return Rso
                
            if Rprocess == "Phenomenological":
                """
                Phenomenological Relaxation
                """
                dim = self.Vdim 
                Rso = R2 * np.ones((dim,dim))
                np.fill_diagonal(Rso, R1) 
                Rso = 2.0 * np.multiply(Rso,rho) 
                return Rso              

            if Rprocess == "Phenomenological Matrix":
                """
                Phenomenological Relaxation
                Relaxation Matrix is given as input
                see function, Relaxation_Phenomenological_Input(R)
                """
                Rso = 2.0 * np.multiply(R_input,rho) 
                return Rso

            if Rprocess == "Auto-correlated Dipolar Heteronuclear":
                """
                Heteronuclear auto-correlated dipolar relaxation (Lindblad, Hilbert space)

                Uses split rank-2 tensor components from Spherical_Tensor_P and their partner components.
                Keeps only auto-correlated pairings.
                Includes the correct adjoint phase consistent with T_{2m}† = (-1)^m T_{2,-m}.
                """

                Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                # Auto-correlated partner pairs ONLY
                pairs = [
                    (10, 10),              # m = 0 (SzSz part)
                    (20, 30), (30, 20),     # m = 0 (flip-flop parts)
                    (11, -11), (-11, 11),   # m = ±1
                    (12, -12), (-12, 12),   # m = ±1
                    (2, -2), (-2, 2),       # m = ±2
                ]

                # Phase that implements the adjoint relation for each label family
                # m=0 -> +1, m=±1 -> -1, m=±2 -> +1
                def phase(label):
                    if label in (11, -11, 12, -12):  # corresponds to |m|=1
                        return -1.0
                    return 1.0
                
                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for a_lbl, b_lbl in pairs:
                        A, wA = self.Spherical_Tensor_P([j, k], 2, a_lbl, Sx, Sy, Sz, Sp, Sm)
                        B, wB = self.Spherical_Tensor_P([j, k], 2, b_lbl, Sx, Sy, Sz, Sp, Sm)

                        Jw = self.SpectralDensity_Lb(wA, self.RelaxParDipole_tau)

                        Rso += (DDC**2) * phase(a_lbl) * Jw * self.Lindblad_Dissipator_Hilbert(A, B, rho)

                Rso = Rso * (-6.0/5.0) * 0.5
                return Rso

            if Rprocess == "Auto-correlated Dipolar Homonuclear":
                """
                Homonuclear auto-correlated dipolar relaxation (Lindblad, Hilbert space)

                Conventions used in PyOR:
                - self.LarmorF[...] is angular frequency ω (rad/s)
                - pairing uses T_{2m} and T_{2,-m}
                - adjoint handled via phase factor: T_{2m}† = (-1)^m T_{2,-m}
                """

                Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]

                for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmn = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        w = m * self.LarmorF[0]  # keep your existing choice (global ω0)
                        Jw = self.SpectralDensity_Lb(w, self.RelaxParDipole_tau)

                        Rso += (DDC**2) * ((-1)**m) * Jw * self.Lindblad_Dissipator_Hilbert(Tm, Tmn, rho)

                Rso = Rso * (-6.0/5.0) * 0.5
                return Rso

            if Rprocess == "Auto-correlated Dipolar Heteronuclear Parallel":
                """
                Heteronuclear auto-correlated dipolar relaxation (Lindblad, Hilbert space),
                parallelized over DipolePairs with joblib.

                Keeps same conventions as non-parallel version:
                - DD_Coupling = 2π * RelaxParDipole_bIS
                - uses SpectralDensity_Lb
                - applies phase factor consistent with your label convention
                - final scaling (-6/5)*0.5
                """

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                pairs = [
                    (10, 10),              # m = 0 (SzSz part)
                    (20, 30), (30, 20),     # m = 0 (flip-flop parts)
                    (11, -11), (-11, 11),   # m = ±1
                    (12, -12), (-12, 12),   # m = ±1
                    (2, -2), (-2, 2),       # m = ±2
                ]

                def Phase(label):
                    # matches your existing Hilbert Lindblad hetero block convention
                    if label in (11, -11, 12, -12):  # |m|=1
                        return -1.0
                    return 1.0

                def OnePairContribution(j, k, DDC):
                    Rloc = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for a_lbl, b_lbl in pairs:
                        A, wA = self.Spherical_Tensor_P([j, k], 2, a_lbl, Sx, Sy, Sz, Sp, Sm)
                        B, _  = self.Spherical_Tensor_P([j, k], 2, b_lbl, Sx, Sy, Sz, Sp, Sm)
                        Jw = self.SpectralDensity_Lb(wA, self.RelaxParDipole_tau)
                        Rloc += (DDC**2) * Phase(a_lbl) * Jw * self.Lindblad_Dissipator_Hilbert(A, B, rho)
                    return Rloc

                if self.ParallelEnabled():
                    with self.Blas1ThreadCtx():
                        parts = Parallel(n_jobs=-1, prefer="processes")(
                            delayed(OnePairContribution)(j, k, DDC)
                            for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling)
                        )
                    Rso = np.sum(parts, axis=0)
                else:
                    Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                        Rso += OnePairContribution(j, k, DDC)

                Rso = Rso * (-6.0/5.0) * 0.5
                return Rso

            if Rprocess == "Auto-correlated Dipolar Homonuclear Parallel":
                """
                Homonuclear auto-correlated dipolar relaxation (Lindblad, Hilbert space),
                parallelized over DipolePairs with joblib.

                Keeps same conventions as non-parallel version:
                - self.LarmorF[0] used as global ω0 (rad/s)
                - DD_Coupling = 2π * RelaxParDipole_bIS
                - uses SpectralDensity_Lb
                - includes (-1)**m factor
                - final scaling (-6/5)*0.5
                """

                Spin_INDEX_1, Spin_INDEX_2 = np.array(self.DipolePairs).T
                DD_Coupling = 2.0 * np.pi * np.asarray(self.RelaxParDipole_bIS)

                mvals = [-2, -1, 0, 1, 2]
                w0 = self.LarmorF[0]

                def OnePairContribution(j, k, DDC):
                    Rloc = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for m in mvals:
                        Tm  = self.Spherical_Tensor([j, k], 2,  m, Sx, Sy, Sz, Sp, Sm)
                        Tmn = self.Spherical_Tensor([j, k], 2, -m, Sx, Sy, Sz, Sp, Sm)

                        w = m * w0
                        Jw = self.SpectralDensity_Lb(w, self.RelaxParDipole_tau)

                        Rloc += (DDC**2) * ((-1)**m) * Jw * self.Lindblad_Dissipator_Hilbert(Tm, Tmn, rho)
                    return Rloc

                if self.ParallelEnabled():
                    with self.Blas1ThreadCtx():
                        parts = Parallel(n_jobs=-1, prefer="processes")(
                            delayed(OnePairContribution)(j, k, DDC)
                            for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling)
                        )
                    Rso = np.sum(parts, axis=0)
                else:
                    Rso = np.zeros((self.Vdim, self.Vdim), dtype=np.cdouble)
                    for j, k, DDC in zip(Spin_INDEX_1, Spin_INDEX_2, DD_Coupling):
                        Rso += OnePairContribution(j, k, DDC)

                Rso = Rso * (-6.0/5.0) * 0.5
                return Rso
        
    def Blas1ThreadCtx(self):
        """
        Context manager to force BLAS/OpenMP-backed NumPy ops to 1 thread.
        Prevents oversubscription when using joblib processes.
        """
        if HAVE_THREADPOOLCTL:
            return threadpool_limits(limits=1)

        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False

        return DummyCtx()

    def ParallelEnabled(self):
        """
        Heuristic: only parallelize when there are enough dipole pairs to amortize overhead.
        Adjust threshold to taste.
        """
        try:
            return len(self.DipolePairs) >= 6
        except Exception:
            return False

