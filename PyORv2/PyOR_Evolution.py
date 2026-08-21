"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This file contains the class `Evolutions`, which is used for simulating 
    time evolution of spin systems under different Hamiltonians and conditions.

Acknowledgements:
    John Price, Q Magnetics, suggestion on Dwell time.
    Marta Stefańska, University of Basel, Biozentrum
"""


import numpy as np
from numpy import linalg as lina
import re
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from IPython.display import display, Latex, Math
from sympy.physics.quantum.cg import CG

try:
    from .PyOR_Relaxation import RelaxationProcess
    from .PyOR_NonlinearNMR import NonLinear
    from .PyOR_QuantumObject import QunObj
    from .PyOR_Commutators import Commutators
except ImportError:
    from PyOR_Relaxation import RelaxationProcess
    from PyOR_NonlinearNMR import NonLinear
    from PyOR_QuantumObject import QunObj
    from PyOR_Commutators import Commutators


class Evolutions:    
    def __init__(self, class_QS,class_Ham=None):
        self.class_QS = class_QS
        self.class_Ham = class_Ham
        self.class_NonL = NonLinear(class_QS)
        self.class_Relax = RelaxationProcess(class_QS)
        self.COMM = Commutators()
        self.PropagationSpace = self.class_QS.PropagationSpace
        self.PropagationMethod = self.class_QS.PropagationMethod
        self.OdeMethod = self.class_QS.OdeMethod
        self.AcqAQ = self.class_QS.AcqAQ
        self.AcqDT = self.class_QS.AcqDT
        #self.Npoints = int(self.AcqAQ/self.AcqDT) # Vineeth
        #self.Npoints = round(self.AcqAQ/self.AcqDT)+1 # John Price
        self.ShapeParOmega = self.class_QS.ShapeParOmega
        self.ShapeParFreq = self.class_QS.ShapeParFreq
        self.ShapeParPhase = self.class_QS.ShapeParPhase
        self.Vdim = self.class_QS.Vdim
        self.Ldim = self.class_QS.Ldim
        self.ODE_atol = self.class_QS.ODE_atol
        self.ODE_rtol = self.class_QS.ODE_rtol
        self.ShapeFunc_or_Hamiltonian = self.class_QS.ShapeFunc_or_Hamiltonian
        self.Lindblad_Temp = self.class_QS.Lindblad_Temp
        self.UserDefined_TimeDependentHamiltonian = self.class_QS.UserDefined_TimeDependentHamiltonian # User-defined function H(t)


    def Update(self):
        self.PropagationSpace = self.class_QS.PropagationSpace
        self.PropagationMethod = self.class_QS.PropagationMethod
        self.OdeMethod = self.class_QS.OdeMethod
        self.AcqAQ = self.class_QS.AcqAQ
        self.AcqDT = self.class_QS.AcqDT
        #self.Npoints = int(self.AcqAQ/self.AcqDT) # Vineeth
        #self.Npoints = round(self.AcqAQ/self.AcqDT)+1 # John Price
        self.ShapeParOmega = self.class_QS.ShapeParOmega
        self.ShapeParFreq = self.class_QS.ShapeParFreq
        self.ShapeParPhase = self.class_QS.ShapeParPhase
        self.Vdim = self.class_QS.Vdim
        self.Ldim = self.class_QS.Ldim
        self.ODE_atol = self.class_QS.ODE_atol
        self.ODE_rtol = self.class_QS.ODE_rtol
        self.ShapeFunc_or_Hamiltonian = self.class_QS.ShapeFunc_or_Hamiltonian
        self.Lindblad_Temp = self.class_QS.Lindblad_Temp

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Time evolution of Density Matrix in Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

    def Set_UserDefinedHamiltonian(self, H):
        """
        Store a user-defined time-dependent Hamiltonian function.

        Parameters
        ----------
        H : callable
            A function with the form

                H(t)

            where t is a scalar time and the returned object is either:

            - a PyOR quantum object with a ``data`` attribute, or
            - a NumPy-compatible square matrix.

        Notes
        -----
        This method stores the function itself. It does not evaluate H(t)
        until TimeDependent_Hamiltonian(t) is called.
        """

        if not callable(H):
            raise TypeError(
                "The user-defined Hamiltonian must be a callable "
                "function with the form H(t)."
            )

        self.UserDefined_TimeDependentHamiltonian = H
        
    def TimeDependent_Hamiltonian(self, t):
        """
        Return the time-dependent Hamiltonian at time t.
        """

        if self.ShapeFunc_or_Hamiltonian == "Off Resonance":

            H_t = self.class_Ham.Zeeman_B1_Offresonance(
                t,
                self.ShapeParOmega,
                -self.ShapeParFreq,
                self.ShapeParPhase,
            )

        elif self.ShapeFunc_or_Hamiltonian == "Bruker":

            H_t = self.class_Ham.Zeeman_B1_ShapedPulse(
                t,
                self.ShapeParOmega,
                -self.ShapeParFreq,
                self.ShapeParPhase,
            )

        elif self.ShapeFunc_or_Hamiltonian == "User Defined Hamiltonian":

            if self.UserDefined_TimeDependentHamiltonian is None:
                raise ValueError(
                    "No user-defined Hamiltonian has been set.\n"
                    "Use Evolutions.Set_UserDefinedHamiltonian(H)."
                )

            H_t = self.UserDefined_TimeDependentHamiltonian(t)

        else:
            raise ValueError(
                f"Unknown ShapeFunc_or_Hamiltonian '{self.ShapeFunc_or_Hamiltonian}'."
            )

        # Convert PyOR QuantumObject to numpy array
        if hasattr(H_t, "data"):
            H_t = H_t.data

        return np.asarray(H_t, dtype=complex)

    def TimeDependent_Hamiltonian_Hilbert(self,t):
        """
        """
        H_shape = np.zeros((t.shape[-1],self.Vdim,self.Vdim),dtype=np.double)
        for i in range(t.shape[-1]):
            H_shape[i] = self.TimeDependent_Hamiltonian(t[i]).real
        return H_shape   

    def Compute_Npoints(self,AQ,DT):
        """
        Docstring for Compute_Npoints
        """
        return round(AQ/DT)+1

    def Compute_Tpoints(self,NPOINTS,DT):
        """
        Docstring for Compute_Tpoints
        """
        return np.linspace(0,DT*(NPOINTS-1),NPOINTS,endpoint=True)

    def Evolution(self, rhoQ, rhoeqQ, HamiltonianQ, RelaxationQ=None, HamiltonianArray=None, CompleteGenerator=False):
        """
        Evolve a state in Schrodinger, Hilbert, or Liouville space.

        Parameters
        ----------
        CompleteGenerator : bool, optional
            Liouville space only. Set True when HamiltonianQ is already the
            complete generator L of d(rho)/dt = L @ rho, for example for a
            combined/multi-system model. Default is False.
        """

        Pmethod = self.PropagationMethod
        ode_method = self.OdeMethod
        dt = self.AcqDT
        #Npoints = int(self.AcqAQ/self.AcqDT) # Vineeth
        #t = np.arange(Npoints) * dt # Vineeth
        Npoints = round(self.AcqAQ/self.AcqDT)+1 # John Price
        t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price
        self.Npoints = Npoints
        

        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_ 
        Sz = self.class_QS.Sz_
        Sp = self.class_QS.Sp_
        Sm = self.class_QS.Sm_ 

        if hasattr(rhoeqQ, 'data'):
            rhoeq = rhoeqQ.data
        else:
            rhoeq = rhoeqQ

        if hasattr(rhoQ, 'data'):
            rho = rhoQ.data
        else:
            rho = rhoQ

        if hasattr(HamiltonianQ, 'data'):
            Hamiltonian = np.asarray(HamiltonianQ.data)
        else:
            Hamiltonian = np.asarray(HamiltonianQ)

        if RelaxationQ is not None:
            if hasattr(RelaxationQ, 'data'):
                Relaxation = np.asarray(RelaxationQ.data)
            else:
                Relaxation = np.asarray(RelaxationQ)
        else:
            Relaxation = None


        if self.PropagationSpace == "Schrodinger":
            if Pmethod == "Unitary Propagator":
                vec_ = rho
                vec_t = [vec_]
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt)
                
                for i in range(Npoints-1):
                    vec_ = np.matmul(U,vec_)
                    vec_t.append(vec_)

            elif Pmethod == "ODE Solver":
                vec_ = rho
                vec_t = []

                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                Lvec = vec_.flatten().astype(complex)  # Ensure it's a 1D complex array

                def vecDOT(t, Lvec, Hamiltonian):
                    return -1j * Hamiltonian @ Lvec  # No need for redundant reshaping

                print("ODE method                    =", ode_method)
                print("ODE absolute tolerance (atol) =", self.ODE_atol)
                print("ODE relative tolerance (rtol) =", self.ODE_rtol)
            
                vecSol = solve_ivp(vecDOT,[0,dt*(Npoints-1)],Lvec,method=self.OdeMethod,t_eval=t,args=(Hamiltonian,), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, vec_sol = vecSol.t, vecSol.y

                for i in range(Npoints):
                    vec_t.append(np.reshape(vec_sol[:,i],(vec_.shape[0],1)))

            else:
                raise ValueError(
                    "Unknown Schrodinger-space propagation method: "
                    f"'{Pmethod}'.\n"
                    "Use 'Unitary Propagator' or 'ODE Solver'."
                )

            return t, vec_t

        if self.PropagationSpace == "Hilbert":

            # ============================================================
            # 1. UNITARY PROPAGATOR
            # ============================================================

            if Pmethod == "Unitary Propagator":

                rho_t = np.zeros(
                    (Npoints, self.Vdim, self.Vdim),
                    dtype=complex
                )

                rho_t[0] = rho

                if HamiltonianArray is not None and len(HamiltonianArray) < (Npoints - 1):
                    raise ValueError(
                        f"HamiltonianArray must contain at least {Npoints - 1} time-step matrices. "
                        f"Received {len(HamiltonianArray)}."
                    )

                HasTimeDependentHamiltonian = (
                    HamiltonianArray is not None
                    or self.ShapeFunc_or_Hamiltonian in (
                        "Off Resonance",
                        "Bruker",
                        "User Defined Hamiltonian"
                    )
                )

                if not HasTimeDependentHamiltonian:

                    U = expm(-1j * Hamiltonian * dt)

                    for i in range(Npoints - 1):
                        rho = U @ rho @ U.conj().T
                        rho_t[i + 1] = rho

                else:

                    for i in range(Npoints - 1):

                        if HamiltonianArray is not None:
                            H = Hamiltonian + HamiltonianArray[i]
                        else:
                            H = Hamiltonian + self.TimeDependent_Hamiltonian(t[i])

                        U = expm(-1j * H * dt)
                        rho = U @ rho @ U.conj().T
                        rho_t[i + 1] = rho

            # ============================================================
            # 2. ODE SOLVER
            # ============================================================

            elif Pmethod == "ODE Solver":

                """
                General Hilbert-space ODE solver.

                Automatically handles:

                - Redfield master equation
                - Lindblad master equation
                - Phenomenological relaxation
                - Phenomenological relaxation matrix
                - Radiation damping
                - Dipolar shift
                - Time-dependent / shaped-pulse Hamiltonian
                """

                rho_t = np.zeros(
                    (Npoints, self.Vdim, self.Vdim),
                    dtype=complex
                )

                # Convert density matrix to vector for solve_ivp
                rhoi = rho.reshape(-1) + 0j


                # ========================================================
                # Differential equation
                # ========================================================

                def rhoDOT(
                    t,
                    rho,
                    rhoeq,
                    Hamiltonian,
                    Sx,
                    Sy,
                    Sz,
                    Sp,
                    Sm
                ):

                    # ----------------------------------------------------
                    # Convert vector back to density matrix
                    # ----------------------------------------------------

                    rho_temp = np.reshape(
                        rho,
                        (self.Vdim, self.Vdim)
                    )


                    # ====================================================
                    # RELAXATION
                    # ====================================================

                    # ----------------------------------------------------
                    # Lindblad master equation
                    #
                    # Relaxation acts directly on rho
                    # ----------------------------------------------------

                    if self.class_Relax.MasterEquation == "Lindblad":

                        # Lindblad relaxation acts directly on rho.
                        Rso_temp = self.class_Relax.Relaxation(
                            rho_temp
                        )

                    else:

                        # Redfield / phenomenological relaxation acts on
                        # the deviation from equilibrium. The relaxation
                        # process is selected through QS.Configure(...).
                        Rso_temp = self.class_Relax.Relaxation(
                            rho_temp - rhoeq
                        )


                    # ====================================================
                    # NONLINEAR TERMS
                    # ====================================================

                    # Radiation damping field
                    Brd = self.class_NonL.Radiation_Damping(
                        rho_temp
                    )

                    # Dipolar field shift
                    Bdipole = self.class_NonL.DipoleShift(
                        rho_temp
                    )


                    # ====================================================
                    # HAMILTONIAN
                    # ====================================================

                    H = (
                        Hamiltonian
                        + np.sum(Sx, axis=0) * Brd.real
                        + np.sum(Sy, axis=0) * Brd.imag
                        + np.sum(Sz, axis=0) * Bdipole
                    )


                    # ====================================================
                    # TIME-DEPENDENT HAMILTONIAN
                    # ====================================================

                    if self.ShapeFunc_or_Hamiltonian in (
                        "Off Resonance",
                        "Bruker",
                        "User Defined Hamiltonian"
                    ):

                        H_shapePulse = self.TimeDependent_Hamiltonian(t)

                        H = H + H_shapePulse


                    # ====================================================
                    # LIOUVILLE-von NEUMANN EQUATION
                    # ====================================================

                    rhodot = (
                        -1j * self.Commutator(H, rho_temp)
                        - Rso_temp
                    )

                    return rhodot.reshape(-1)


                # ========================================================
                # SOLVE DIFFERENTIAL EQUATION
                # ========================================================

                print("ODE method                    =", ode_method)
                print("ODE absolute tolerance (atol) =", self.ODE_atol)
                print("ODE relative tolerance (rtol) =", self.ODE_rtol)

                rhoSol = solve_ivp(
                    rhoDOT,
                    [0, dt * (Npoints - 1)],
                    rhoi,
                    method=ode_method,
                    t_eval=t,
                    args=(
                        rhoeq,
                        Hamiltonian,
                        Sx,
                        Sy,
                        Sz,
                        Sp,
                        Sm
                    ),
                    atol=self.ODE_atol,
                    rtol=self.ODE_rtol
                )


                # ========================================================
                # CONVERT SOLUTION BACK TO DENSITY MATRICES
                # ========================================================

                t, rho2d = rhoSol.t, rhoSol.y

                for i in range(Npoints):

                    rho = np.reshape(
                        rho2d[:, i],
                        (self.Vdim, self.Vdim)
                    )

                    rho_t[i] = rho


            # ============================================================
            # UNKNOWN PROPAGATION METHOD
            # ============================================================

            else:

                raise ValueError(
                    "Unknown Hilbert-space propagation method: "
                    f"'{Pmethod}'.\n"
                    "Use 'Unitary Propagator' or 'ODE Solver'."
                )

            return t, rho_t
        
        if self.PropagationSpace == "Liouville":
            """
            Evolution in Liouville space.

            Only two propagation methods are used:

                "Unitary Propagator"
                "ODE Solver"

            The selected master equation, relaxation process,
            sparse/dense matrices and time-dependent Hamiltonian
            are handled automatically.
            """

            # ============================================================
            # Basic dimensions
            # ============================================================

            SystemDim = rho.shape[0]

            # Make sure rho is a column vector
            if rho.ndim == 1:
                rho = rho.reshape(SystemDim, 1)

            # Make sure rhoeq is also a column vector
            if rhoeq is not None and rhoeq.ndim == 1:
                rhoeq = rhoeq.reshape(SystemDim, 1)


            # ============================================================
            # Determine system type
            # ============================================================

            # Normal PyOR Liouville evolution uses the usual
            # -1j * Hamiltonian convention.
            #
            # For a combined/multi-system model, set CompleteGenerator=True.
            # In that case HamiltonianQ is interpreted as the COMPLETE generator L:
            #
            #       d(rho)/dt = L @ rho
            #
            # and PyOR must NOT multiply it by -1j again.
            CombinedSystem = bool(CompleteGenerator)
            NormalSystem = not CombinedSystem

            if NormalSystem and SystemDim != self.Ldim:
                raise ValueError(
                    f"Liouville state dimension is {SystemDim}, but the native PyOR "
                    f"Liouville dimension is {self.Ldim}. If HamiltonianQ is an already "
                    "assembled complete generator for a combined/multi-system model, "
                    "call Evolution(..., CompleteGenerator=True)."
                )


            # ============================================================
            # Determine master equation
            # ============================================================

            Lindblad = (
                self.class_Relax.MasterEquation == "Lindblad"
            )


            # ============================================================
            # Determine whether relaxation is present
            # ============================================================

            HasRelaxation = RelaxationQ is not None


            # ============================================================
            # Determine whether a time-dependent Hamiltonian exists
            # ============================================================

            if HamiltonianArray is not None and len(HamiltonianArray) < (Npoints - 1):
                raise ValueError(
                    f"HamiltonianArray must contain at least {Npoints - 1} time-step matrices. "
                    f"Received {len(HamiltonianArray)}."
                )


            HasTimeDependentHamiltonian = (

                HamiltonianArray is not None

                or

                getattr(
                    self,
                    "ShapeFunc_or_Hamiltonian",
                    None
                ) in (
                    "Off Resonance",
                    "Bruker",
                    "User Defined Hamiltonian"
                )
            )


            # ============================================================
            # Function for obtaining H(t)
            # ============================================================

            def GetHamiltonian(ti, i=None):

                H = Hamiltonian

                # Hamiltonian supplied as an array
                if HamiltonianArray is not None:

                    if i is None:
                        i = int(np.clip(np.searchsorted(t, ti, side="right") - 1, 0, len(HamiltonianArray) - 1))

                    H = H + HamiltonianArray[i]

                # Hamiltonian supplied as a function
                elif getattr(
                    self,
                    "ShapeFunc_or_Hamiltonian",
                    None
                ) in (
                    "Off Resonance",
                    "Bruker",
                    "User Defined Hamiltonian"
                ):

                    H = H + self.TimeDependent_Hamiltonian(ti)

                return H


            # ============================================================
            # 1. UNITARY PROPAGATOR
            # ============================================================

            if Pmethod == "Unitary Propagator":

                rho_t = np.zeros(
                    (Npoints, SystemDim, 1),
                    dtype=complex
                )

                rho_t[0] = rho


                # ========================================================
                # COMBINED / MULTI-SYSTEM
                # ========================================================
                #
                # For the combined system, Hamiltonian is already the
                # complete generator:
                #
                #       dρ/dt = L ρ
                #
                # therefore:
                #
                #       U = exp(L dt)
                #
                # NOT exp(-i L dt)
                # ========================================================

                if CombinedSystem:

                    if RelaxationQ is not None:
                        raise ValueError(
                            "For a combined/multi-system Liouville state, HamiltonianQ "
                            "is treated as the complete generator L. Pass relaxation "
                            "inside that generator and use RelaxationQ=None."
                        )

                    # Check generator dimension
                    if Hamiltonian.shape != (
                        SystemDim,
                        SystemDim
                    ):

                        raise ValueError(
                            "Combined-system generator dimension "
                            "does not match the state vector.\n"
                            f"Generator shape = {Hamiltonian.shape}\n"
                            f"State shape = {rho.shape}"
                        )


                    # ----------------------------------------------------
                    # Time-independent generator
                    # ----------------------------------------------------

                    if not HasTimeDependentHamiltonian:

                        U = expm(
                            Hamiltonian * dt
                        )

                        for i in range(Npoints - 1):

                            rho = U @ rho

                            rho_t[i + 1] = rho


                    # ----------------------------------------------------
                    # Time-dependent generator
                    # ----------------------------------------------------

                    else:

                        for i in range(Npoints - 1):

                            L = GetHamiltonian(
                                t[i],
                                i
                            )

                            rho = expm_multiply(
                                L * dt,
                                rho
                            )

                            rho_t[i + 1] = rho


                # ========================================================
                # NORMAL PyOR LIOUVILLE SYSTEM
                # ========================================================

                else:

                    # ====================================================
                    # No relaxation
                    # ====================================================

                    if not HasRelaxation:

                        # ------------------------------------------------
                        # Time-independent Hamiltonian
                        # ------------------------------------------------

                        if not HasTimeDependentHamiltonian:

                            # Sparse Hamiltonian
                            if sparse.issparse(Hamiltonian):

                                U = sparse.linalg.expm(
                                    -1j * Hamiltonian * dt
                                )

                            # Dense Hamiltonian
                            else:

                                U = expm(
                                    -1j * Hamiltonian * dt
                                )


                            for i in range(Npoints - 1):

                                rho = U @ rho

                                rho_t[i + 1] = rho


                        # ------------------------------------------------
                        # Time-dependent Hamiltonian
                        # ------------------------------------------------

                        else:

                            for i in range(Npoints - 1):

                                H = GetHamiltonian(
                                    t[i],
                                    i
                                )

                                A = -1j * H * dt

                                rho = expm_multiply(
                                    A,
                                    rho
                                )

                                rho_t[i + 1] = rho


                    # ====================================================
                    # Relaxation present
                    # ====================================================

                    else:

                        # ------------------------------------------------
                        # Time-independent Hamiltonian + relaxation
                        # ------------------------------------------------

                        if not HasTimeDependentHamiltonian:

                            A = (
                                -1j * Hamiltonian
                                - Relaxation
                            ) * dt


                            # Sparse matrix
                            if sparse.issparse(A):

                                U = sparse.linalg.expm(A)

                            # Dense matrix
                            else:

                                U = expm(A)


                            # --------------------------------------------
                            # Lindblad
                            #
                            #     rho(t+dt) = U rho(t)
                            # --------------------------------------------

                            if Lindblad:

                                for i in range(Npoints - 1):

                                    rho = U @ rho

                                    rho_t[i + 1] = rho


                            # --------------------------------------------
                            # Redfield / equilibrium relaxation
                            #
                            # rho(t+dt)
                            # =
                            # U [rho(t)-rhoeq] + rhoeq
                            # --------------------------------------------

                            else:

                                for i in range(Npoints - 1):

                                    rho = (
                                        U @ (rho - rhoeq)
                                        + rhoeq
                                    )

                                    rho_t[i + 1] = rho


                        # ------------------------------------------------
                        # Time-dependent Hamiltonian + relaxation
                        # ------------------------------------------------

                        else:

                            for i in range(Npoints - 1):

                                H = GetHamiltonian(
                                    t[i],
                                    i
                                )

                                A = (
                                    -1j * H
                                    - Relaxation
                                ) * dt


                                # ----------------------------------------
                                # Lindblad
                                # ----------------------------------------

                                if Lindblad:

                                    rho = expm_multiply(
                                        A,
                                        rho
                                    )


                                # ----------------------------------------
                                # Redfield
                                # ----------------------------------------

                                else:

                                    rho = (
                                        expm_multiply(
                                            A,
                                            rho - rhoeq
                                        )
                                        + rhoeq
                                    )


                                rho_t[i + 1] = rho


            # ============================================================
            # 2. ODE SOLVER
            # ============================================================

            elif Pmethod == "ODE Solver":

                rho_t = np.zeros(
                    (Npoints, SystemDim, 1),
                    dtype=complex
                )

                print("ODE method                    =", ode_method)
                print("ODE absolute tolerance (atol) =", self.ODE_atol)
                print("ODE relative tolerance (rtol) =", self.ODE_rtol)


                # ========================================================
                # Convert column vector to 1D vector for solve_ivp
                # ========================================================

                Lrho = rho.reshape(SystemDim) + 0j

                if rhoeq is not None:

                    Lrhoeq = rhoeq.reshape(SystemDim)

                else:

                    Lrhoeq = np.zeros(
                        SystemDim,
                        dtype=complex
                    )


                # ========================================================
                # COMBINED / MULTI-SYSTEM ODE
                # ========================================================

                if CombinedSystem:

                    if RelaxationQ is not None:
                        raise ValueError(
                            "For a combined/multi-system Liouville state, HamiltonianQ "
                            "is treated as the complete generator L. Pass relaxation "
                            "inside that generator and use RelaxationQ=None."
                        )

                    def rhoDOT(ti, Lrho):

                        # Base generator
                        L = Hamiltonian


                        # -----------------------------------------------
                        # Hamiltonian array
                        # -----------------------------------------------

                        if HamiltonianArray is not None:

                            index = np.searchsorted(
                                t,
                                ti
                            )

                            index = min(
                                index,
                                len(HamiltonianArray) - 1
                            )

                            L = (
                                L
                                + HamiltonianArray[index]
                            )


                        # -----------------------------------------------
                        # User-defined time-dependent generator
                        # -----------------------------------------------

                        elif getattr(
                            self,
                            "ShapeFunc_or_Hamiltonian",
                            None
                        ) in (
                            "Off Resonance",
                            "Bruker",
                            "User Defined Hamiltonian"
                        ):

                            L = (
                                L
                                + self.TimeDependent_Hamiltonian(ti)
                            )


                        # Complete generator already supplied
                        rhodot = L @ Lrho

                        return np.asarray(
                            rhodot
                        ).reshape(-1)


                    rhoSol = solve_ivp(

                        rhoDOT,

                        [
                            0,
                            dt * (Npoints - 1)
                        ],

                        Lrho,

                        method=ode_method,

                        t_eval=t,

                        atol=self.ODE_atol,

                        rtol=self.ODE_rtol
                    )


                # ========================================================
                # NORMAL PyOR LIOUVILLE ODE
                # ========================================================

                else:

                    def rhoDOT(
                        ti,
                        Lrho,
                        LHamiltonian,
                        RsuperOP,
                        Lrhoeq,
                        Sx,
                        Sy
                    ):

                        # ================================================
                        # Convert Liouville vector to density matrix
                        # ================================================

                        rho_temp = self.Convert_LrhoTO2Drho(
                            Lrho
                        )


                        # ================================================
                        # Radiation damping
                        # ================================================

                        Brd = self.class_NonL.Radiation_Damping(
                            rho_temp
                        )


                        # ================================================
                        # Hamiltonian
                        # ================================================

                        LH = (
                            LHamiltonian
                            + self.CommutationSuperoperator(
                                np.sum(Sx, axis=0)
                                * Brd.real
                            )
                            + self.CommutationSuperoperator(
                                np.sum(Sy, axis=0)
                                * Brd.imag
                            )
                        )


                        # ================================================
                        # Time-dependent Hamiltonian
                        # ================================================

                        if HamiltonianArray is not None:

                            index = np.searchsorted(
                                t,
                                ti
                            )

                            index = min(
                                index,
                                len(HamiltonianArray) - 1
                            )

                            LH = (
                                LH
                                + HamiltonianArray[index]
                            )


                        elif getattr(
                            self,
                            "ShapeFunc_or_Hamiltonian",
                            None
                        ) in (
                            "Off Resonance",
                            "Bruker",
                            "User Defined Hamiltonian"
                        ):

                            LH = (
                                LH
                                + self.TimeDependent_Hamiltonian(ti)
                            )


                        # ================================================
                        # No relaxation
                        # ================================================

                        if RsuperOP is None:

                            rhodot = (
                                -1j * LH @ Lrho
                            )


                        # ================================================
                        # Lindblad relaxation
                        #
                        #     dρ/dt = -i Lρ - Rρ
                        # ================================================

                        elif Lindblad:

                            rhodot = (
                                -1j * LH @ Lrho
                                - RsuperOP @ Lrho
                            )


                        # ================================================
                        # Redfield relaxation
                        #
                        # dρ/dt =
                        #
                        # -i Lρ
                        #
                        # -R(ρ-rhoeq)
                        # ================================================

                        else:

                            rhodot = (
                                -1j * LH @ Lrho
                                - RsuperOP @ (
                                    Lrho - Lrhoeq
                                )
                            )


                        return np.asarray(
                            rhodot
                        ).reshape(-1)


                    # ====================================================
                    # Solve ODE
                    # ====================================================

                    rhoSol = solve_ivp(

                        rhoDOT,

                        [
                            0,
                            dt * (Npoints - 1)
                        ],

                        Lrho,

                        method=ode_method,

                        t_eval=t,

                        args=(
                            Hamiltonian,
                            Relaxation,
                            Lrhoeq,
                            Sx,
                            Sy
                        ),

                        atol=self.ODE_atol,

                        rtol=self.ODE_rtol
                    )


                # ========================================================
                # Store solution
                # ========================================================

                t, rho_sol = (
                    rhoSol.t,
                    rhoSol.y
                )

                for i in range(len(t)):

                    rho_t[i] = np.reshape(
                        rho_sol[:, i],
                        (SystemDim, 1)
                    )


            # ============================================================
            # Invalid method
            # ============================================================

            else:

                raise ValueError(
                    "Unknown Liouville-space propagation method: "
                    f"'{Pmethod}'.\n"
                    "Use 'Unitary Propagator' or 'ODE Solver'."
                )


            return t, rho_t
            
    def Expectation(self,rho_t,detectionQ, tolerance=1.0e-14):

        dt = self.AcqDT

        #Npoints = int(self.AcqAQ/self.AcqDT) # Vineeth
        #t = np.arange(Npoints) * dt # Vineeth
        ##t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth

        Npoints = round(self.AcqAQ/self.AcqDT)+1 # John Price
        t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

        detection = detectionQ.data

        if self.PropagationSpace == "Schrodinger":
            
            signal = np.zeros(Npoints,dtype=complex)

            for i in range(Npoints):
                signal[i] = np.trace(np.matmul(rho_t[i].conj().T,np.matmul(detection,rho_t[i]))) 

            signal[np.abs(signal) < tolerance] = 0.0 + 0.0j

            return t, signal                   
    
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

            for i in range(Npoints):
                #signal[i] = np.trace(np.matmul(detection,rho_t[i]))
                signal[i] = np.trace(np.matmul(rho_t[i],detection))

            signal[np.abs(signal) < tolerance] = 0.0 + 0.0j

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

            for i in range(Npoints):
                signal[i] = np.trace(detection.T @ rho_t[i])

            signal[np.abs(signal) < tolerance] = 0.0 + 0.0j

            return t, signal   

    def PartialTrace_(self, rho, keep, Sdim = None):
        """
        Compute the partial trace over specified subsystems of a density matrix.

        Parameters
        ----------
        rho : QunObj
            Density matrix.
        keep : list of int
            Indices of subsystems to retain.
        Sdim : list of int, optional
            Dimensions of subsystems (defaults to class_QS.Sdim).

        Returns
        -------
        QunObj
            Reduced density matrix after tracing out unlisted subsystems.
        """

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()

        SysInx = range(len(Sdim))
        TraceInx = list(set(SysInx) - set(keep))

        new_shape = Sdim + Sdim
        rho_new = rho.reshape(new_shape)

        for idx in sorted(TraceInx, reverse=True):
            rho_new = np.trace(rho_new, axis1=idx, axis2=idx + len(Sdim))

        Sdim_new = [Sdim[j] for j in keep]
        final_shape = (np.prod(Sdim_new), np.prod(Sdim_new))

        return rho_new.reshape(final_shape)      

    def PartialTrace(self, rho, keep, Sdim = None):
        """
        Compute the partial trace over specified subsystems of a density matrix.

        Parameters
        ----------
        rho : QunObj
            Density matrix.
        keep : list of int
            Indices of subsystems to retain.
        Sdim : list of int, optional
            Dimensions of subsystems (defaults to class_QS.Sdim).

        Returns
        -------
        QunObj
            Reduced density matrix after tracing out unlisted subsystems.

        Bug fix by Marta Stefańska    
        """

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()

        SysInx = range(len(Sdim))
        TraceInx = list(set(SysInx) - set(keep))

        Sdim_current = list(Sdim)  # MODIFIED
        rho_new = rho.copy()  # MODIFIED

        for idx in sorted(TraceInx, reverse=True):
            n_curr = len(Sdim_current)  # ADDED
            rho_new = np.trace(rho_new.reshape(Sdim_current + Sdim_current),  # MODIFIED
                               axis1=idx, axis2=idx + n_curr)  # MODIFIED
            Sdim_current.pop(idx)  # ADDED

        Sdim_new = [Sdim[j] for j in keep]
        final_shape = (np.prod(Sdim_new), np.prod(Sdim_new))

        return rho_new.reshape(final_shape)

    def PartialTrace_Vector(self, rho_vec, keep, Sdim=None):
        """
        Compute the partial trace directly on a vectorized density matrix.

        Parameters
        ----------
        rho_vec : ndarray
            Vectorized density matrix with shape (Ldim,),
            (Ldim, 1), or (1, Ldim).

        keep : list of int
            Indices of subsystems to retain, using 0-based indexing.

        Sdim : list of int, optional
            Dimensions of the Hilbert-space subsystems.
            Defaults to self.class_QS.Sdim.

        Returns
        -------
        ndarray
            Vectorized reduced density matrix with shape
            (Ldim_reduced, 1).
        """

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()

        Sdim = list(Sdim)
        keep = list(keep)

        n_subsystems = len(Sdim)
        Vdim = int(np.prod(Sdim))
        Ldim = Vdim**2

        rho_vec = np.asarray(rho_vec).reshape(-1)

        if rho_vec.size != Ldim:
            raise ValueError(
                f"Input contains {rho_vec.size} elements, "
                f"but expected {Ldim} for Sdim={Sdim}."
            )

        if len(set(keep)) != len(keep):
            raise ValueError("The keep list contains duplicate indices.")

        if not set(keep).issubset(range(n_subsystems)):
            raise ValueError(
                f"Invalid keep indices {keep} for {n_subsystems} subsystems."
            )

        # Directly reshape the Liouville vector into:
        #
        # ket indices: i0, i1, ..., iN
        # bra indices: j0, j1, ..., jN
        #
        rho_tensor = rho_vec.reshape(Sdim + Sdim, order="C")

        trace_indices = sorted(
            set(range(n_subsystems)) - set(keep),
            reverse=True
        )

        Sdim_current = Sdim.copy()

        for idx in trace_indices:
            n_current = len(Sdim_current)

            rho_tensor = np.trace(
                rho_tensor,
                axis1=idx,
                axis2=idx + n_current
            )

            Sdim_current.pop(idx)

        Vdim_reduced = int(np.prod(Sdim_current))

        return rho_tensor.reshape(
            (Vdim_reduced**2, 1),
            order="C"
        )

    def PartialTrace_Vector_Future(self, state, keep, Sdim=None):
        """
        Partial trace for either:
        1. pure state vector
        2. vectorized density matrix

        Returns a vectorized reduced density matrix.
        """

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()

        Sdim = list(Sdim)
        keep = list(keep)

        Vdim = int(np.prod(Sdim))
        Ldim = Vdim**2

        state = np.asarray(state).reshape(-1)

        SysInx = range(len(Sdim))
        TraceInx = sorted(
            set(SysInx) - set(keep),
            reverse=True
        )

        # ==========================================================
        # PURE STATE VECTOR
        # ==========================================================

        if state.size == Vdim:

            psi_tensor = state.reshape(Sdim, order="C")

            trace_axes = sorted(
                set(range(len(Sdim))) - set(keep)
            )

            rho_red = np.tensordot(
                psi_tensor,
                psi_tensor.conj(),
                axes=(trace_axes, trace_axes)
            )

            Sdim_new = [Sdim[i] for i in keep]
            Vdim_new = int(np.prod(Sdim_new))

            return rho_red.reshape(
                (Vdim_new**2, 1),
                order="C"
            )

        # ==========================================================
        # VECTORIZED DENSITY MATRIX
        # ==========================================================

        elif state.size == Ldim:

            rho_tensor = state.reshape(
                Sdim + Sdim,
                order="C"
            )

            Sdim_current = Sdim.copy()

            for idx in TraceInx:

                n_current = len(Sdim_current)

                rho_tensor = np.trace(
                    rho_tensor,
                    axis1=idx,
                    axis2=idx + n_current
                )

                Sdim_current.pop(idx)

            Vdim_new = int(np.prod(Sdim_current))

            return rho_tensor.reshape(
                (Vdim_new**2, 1),
                order="C"
            )

        else:

            raise ValueError(
                f"Input size {state.size} is invalid. "
                f"Expected {Vdim} for a pure state vector "
                f"or {Ldim} for a vectorized density matrix."
            )

    def PartialTrace_Superoperator(self, keep, Sdim=None, QuantumObject=True):
        """
        Construct the partial-trace superoperator in Liouville space.

        The superoperator Tr satisfies

            vec(rho_reduced) = Tr @ vec(rho)

        Parameters
        ----------
        keep : list of int
            Indices of subsystems to retain.

        Sdim : list of int, optional
            Dimensions of the subsystems.
            Defaults to self.class_QS.Sdim.

        QuantumObject : bool, optional
            If True, return QunObj.
            If False, return NumPy array.

        Returns
        -------
        QunObj or ndarray
            Partial-trace superoperator with shape

            (Ldim_reduced, Ldim_full)

        Example
        -------
        For NH4 -> NH3:

            Tr = QS.B.Evolutions.PartialTraceSuperoperator(
                keep=[0, 1, 2, 3]
            )

        giving shape:

            (256, 1024)
        """

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()
        else:
            Sdim = list(Sdim)

        # Full Hilbert-space dimension
        D_in = int(np.prod(Sdim))

        # Dimensions of retained subsystems
        Sdim_out = [Sdim[i] for i in keep]

        # Reduced Hilbert-space dimension
        D_out = int(np.prod(Sdim_out))

        # Liouville dimensions
        Ldim_in = D_in ** 2
        Ldim_out = D_out ** 2

        # Use the same vectorization convention as PyOR
        order = self.class_QS.RowColOrder

        # Partial-trace superoperator
        Tr = np.zeros(
            (Ldim_out, Ldim_in),
            dtype=self.class_QS.DTYPE_C,
            order=self.class_QS.ORDER_MEMORY
        )

        # Apply PartialTrace to every Liouville basis vector
        for k in range(Ldim_in):

            # Liouville basis vector
            e = np.zeros(
                Ldim_in,
                dtype=self.class_QS.DTYPE_C
            )

            e[k] = 1.0

            # Convert Liouville vector -> matrix
            rho = e.reshape(
                D_in,
                D_in,
                order=order
            )

            # Existing PyOR partial trace
            rho_red = self.PartialTrace(
                rho,
                keep=keep,
                Sdim=Sdim
            )

            # Reduced density matrix -> Liouville vector
            Tr[:, k] = rho_red.reshape(
                -1,
                order=order
            )

        if QuantumObject:
            return QunObj(Tr)

        return Tr

    def Kronecker_Superoperator(self, rho_add, insert_at, Sdim=None, QuantumObject=True):
        """
        Construct a superoperator that inserts a new subsystem
        with density matrix rho_add.

        The superoperator K satisfies

            vec(rho_new) = K @ vec(rho_old)

        Parameters
        ----------
        rho_add : QunObj or ndarray
            Density matrix of the subsystem being added.

        insert_at : int
            Position where the new subsystem is inserted.

        Sdim : list of int, optional
            Dimensions of the original system.
            Defaults to self.class_QS.Sdim.

        QuantumObject : bool, optional
            If True, return QunObj.
            If False, return ndarray.

        Returns
        -------
        QunObj or ndarray
            Kronecker-product superoperator.
        """

        if Sdim is None:
            Sdim = list(self.class_QS.Sdim)
        else:
            Sdim = list(Sdim)

        if hasattr(rho_add, "data"):
            rho_add = rho_add.data

        rho_add = np.asarray(rho_add)

        D_in = int(np.prod(Sdim))

        d_add = rho_add.shape[0]

        Sdim_out = Sdim.copy()
        Sdim_out.insert(insert_at, d_add)

        D_out = int(np.prod(Sdim_out))

        Ldim_in = D_in ** 2
        Ldim_out = D_out ** 2

        order = self.class_QS.RowColOrder

        Kron = np.zeros(
            (Ldim_out, Ldim_in),
            dtype=self.class_QS.DTYPE_C
        )

        for k in range(Ldim_in):

            # Liouville basis vector
            e = np.zeros(
                Ldim_in,
                dtype=self.class_QS.DTYPE_C
            )

            e[k] = 1.0

            # Vector -> operator
            rho = e.reshape(
                D_in,
                D_in,
                order=order
            )

            # Tensor representation of original operator
            rho_tensor = rho.reshape(
                Sdim + Sdim
            )

            # Add new subsystem
            rho_new = np.tensordot(
                rho_tensor,
                rho_add,
                axes=0
            )

            N = len(Sdim)

            # Current ordering after tensordot:
            #
            # old ket axes,
            # old bra axes,
            # new ket,
            # new bra

            ket_axes = list(range(N))
            bra_axes = list(range(N, 2 * N))

            new_ket = 2 * N
            new_bra = 2 * N + 1

            ket_axes.insert(insert_at, new_ket)
            bra_axes.insert(insert_at, new_bra)

            rho_new = np.transpose(
                rho_new,
                ket_axes + bra_axes
            )

            rho_new = rho_new.reshape(
                D_out,
                D_out
            )

            Kron[:, k] = rho_new.reshape(
                -1,
                order=order
            )

        if QuantumObject:
            return QunObj(Kron)

        return Kron

    def Convert_LrhoTO2Drho_(self,Lrho): 
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

    def Convert_LrhoTO2Drho(self, Lrho):
        """
        Convert Liouville-space density vector(s) into Hilbert-space matrices.

        Accepted shapes
        ---------------
        Single state:
            (Ldim,)
            (Ldim, 1)
            (1, Ldim)

        Multiple states:
            (Npoints, Ldim)
            (Ldim, Npoints)
            (Npoints, Ldim, 1)
            (Npoints, 1, Ldim)

        Already converted:
            (Npoints, Vdim, Vdim)
        """

        Lrho = np.asarray(Lrho)

        Vdim = self.Vdim
        Ldim = Vdim**2

        # Single one-dimensional Liouville vector
        if Lrho.ndim == 1:
            if Lrho.size != Ldim:
                raise ValueError(
                    f"Expected {Ldim} elements, received {Lrho.size}."
                )

            return Lrho.reshape(Vdim, Vdim)

        # Two-dimensional inputs
        if Lrho.ndim == 2:

            # Single column or row vector
            if Lrho.shape in ((Ldim, 1), (1, Ldim)):
                return Lrho.reshape(Vdim, Vdim)

            # Each row is one Liouville vector
            if Lrho.shape[1] == Ldim:
                return Lrho.reshape(
                    Lrho.shape[0],
                    Vdim,
                    Vdim
                )

            # Each column is one Liouville vector
            if Lrho.shape[0] == Ldim:
                return Lrho.T.reshape(
                    Lrho.shape[1],
                    Vdim,
                    Vdim
                )

        # Three-dimensional inputs
        if Lrho.ndim == 3:

            # Already converted density-matrix trajectory
            if Lrho.shape[-2:] == (Vdim, Vdim):
                return Lrho

            # Shape: (Npoints, Ldim, 1)
            if Lrho.shape[1:] == (Ldim, 1):
                return Lrho[:, :, 0].reshape(
                    Lrho.shape[0],
                    Vdim,
                    Vdim
                )

            # Shape: (Npoints, 1, Ldim)
            if Lrho.shape[1:] == (1, Ldim):
                return Lrho[:, 0, :].reshape(
                    Lrho.shape[0],
                    Vdim,
                    Vdim
                )

        raise ValueError(
            "Unsupported Lrho shape. Expected one of: "
            f"({Ldim},), ({Ldim}, 1), (1, {Ldim}), "
            f"(Npoints, {Ldim}), ({Ldim}, Npoints), "
            f"(Npoints, {Ldim}, 1), (Npoints, 1, {Ldim}), "
            f"or (Npoints, {Vdim}, {Vdim}). "
            f"Received {Lrho.shape}."
        )

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