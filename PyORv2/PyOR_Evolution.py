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
        
    def TimeDependent_Hamiltonian_(self,t):
        """
        """
        
        if self.ShapeFunc == "Off Resonance":
            return self.class_Ham.Zeeman_B1_Offresonance(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)
        if self.ShapeFunc == "Bruker":
            return self.class_Ham.Zeeman_B1_ShapedPulse(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)        

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
            raise ValueError(f"Unknown ShapeFunc '{self.ShapeFunc}'.")

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

    def Evolution(self,rhoQ,rhoeqQ,HamiltonianQ,RelaxationQ=None,HamiltonianArray=None):

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

        Hamiltonian = np.array(HamiltonianQ.data)

        if RelaxationQ is not None:
            Relaxation = np.array(RelaxationQ.data)
        else:
            Relaxation = np.zeros_like(Hamiltonian)  # Ensures Relaxation is always defined


        if self.PropagationSpace == "Schrodinger":
            if Pmethod == "Unitary Propagator":
                vec_ = rhoQ.data
                vec_t = [vec_]
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt)
                
                for i in range(Npoints-1):
                    vec_ = np.matmul(U,vec_)
                    vec_t.append(vec_)

            if Pmethod == "ODE Solver":
                vec_ = rhoQ.data
                vec_t = []

                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                Lvec = vec_.flatten().astype(complex)  # Ensure it's a 1D complex array

                def vecDOT(t, Lvec, Hamiltonian):
                    return -1j * Hamiltonian @ Lvec  # No need for redundant reshaping
            
                vecSol = solve_ivp(vecDOT,[0,dt*(Npoints-1)],Lvec,method=self.OdeMethod,t_eval=t,args=(Hamiltonian,), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, vec_sol = vecSol.t, vecSol.y

                for i in range(Npoints):
                    vec_t.append(np.reshape(vec_sol[:,i],(vec_.shape[0],1)))

            return t, vec_t


        if self.PropagationSpace == "Hilbert":
            
            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,np.matmul(rho,U.T.conj()))
                    rho_t[i+1] = rho   

            if Pmethod == "Unitary Propagator Time Dependent":    
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rho_t[0] = rho
                for i in range(Npoints-1):
                    U = expm(-1j * (Hamiltonian + HamiltonianArray[i]) * dt)
                    rho = np.matmul(U,np.matmul(rho,U.T.conj()))
                    rho_t[i+1] = rho

            if Pmethod == "ODE Solver":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex) 
                #t = np.arange(Npoints) * dt # Vineeth                      
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) #Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rso_temp = self.class_Relax.Relaxation(rho_temp-rhoeq)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho	            

            if Pmethod == "ODE Solver Lindblad":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)       
                #t = np.arange(Npoints) * dt # Vineeth                
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,Hamiltonian,Sx,Sy,Sz,Sp,Sm):                    
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))                       
                    Rso_temp = self.class_Relax.Relaxation(rho_temp)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho

            if Pmethod == "ODE Solver ShapedPulse Lindblad":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)       
                #t = np.arange(Npoints) * dt # Vineeth                
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,Hamiltonian,Sx,Sy,Sz,Sp,Sm):                    
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))                       
                    Rso_temp = self.class_Relax.Relaxation(rho_temp)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H_shapePulse = self.TimeDependent_Hamiltonian(t)
                    H = H_shapePulse + Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
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
                #t = np.arange(Npoints) * dt # Vineeth                    
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rso_temp = self.class_Relax.Relaxation(rho_temp-rhoeq)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    #Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H_shapePulse = self.TimeDependent_Hamiltonian(t)
                    #H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole + H_shapePulse 
                    H = H_shapePulse + Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
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
                #t = np.arange(Npoints) * dt # Vineeth                  
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    Rprocess2 = "Phenomenological Matrix"
                    Rso_temp = self.class_Relax.Relaxation(rho_temp-rhoeq) + self.class_Relax.Relaxation(rho_temp-rhoeq,Rprocess2)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
                t, rho2d = rhoSol.t, rhoSol.y
                for i in range(Npoints):          
                    rho = np.reshape(rho2d[:,i],(self.Vdim,self.Vdim))
                    rho_t[i] = rho

            if Pmethod == "ODE Solver Lindblad Relaxation and Phenomenological":
                """
                Relaxation possible in Hilbert space by using solver for ODE. 
                Integrators not supported: 'Radau' and LSODA
                """
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)     
                #t = np.arange(Npoints) * dt # Vineeth                  
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))

                    Rprocess2 = "Phenomenological Matrix"
                    Rso_temp = self.class_Relax.Relaxation(rho_temp) + self.class_Relax.Relaxation(rho_temp,Rprocess2)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
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
                #t = np.arange(Npoints) * dt # Vineeth                      
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                rhoi = (rho.reshape(-1))
                rho_RI = np.zeros((2*rhoi.shape[-1]))
                rho_RI[0::2] = rhoi.real
                rho_RI[1::2] = rhoi.imag
                
                def rhoDOT(t,rho,rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm):
                    rho = np.reshape(rho[0::2] + 1j * rho[1::2],(self.Vdim,self.Vdim))
                    rhodot = np.zeros((2*rhoi.shape[-1]))
                    Rso = self.class_Relax.Relaxation(rho-rhoeq)
                    Brd = self.class_NonL.Radiation_Damping(rho)                 
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag        
                    rhodot[0::2] = (-1j * self.Commutator(H,rho) - Rso).reshape(-1).real  
                    rhodot[1::2] = (-1j * self.Commutator(H,rho) - Rso).reshape(-1).imag     
                    return rhodot 
                
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],rho_RI,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)

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

            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho)  
                    rho_t[i+1] = rho  

            if Pmethod == "Unitary Propagator Sparse":  
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = sparse.linalg.expm(-1j * Hamiltonian * dt) # LHamiltonian is sparse matrix
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = U.dot(rho)  
                    rho_t[i+1] = rho
            
            if Pmethod == "Relaxation":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho - rhoeq) + rhoeq
                    rho_t[i+1] = rho        

            if Pmethod == "Relaxation Sparse":   
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = sparse.linalg.expm(-1j * Hamiltonian * dt - Relaxation * dt) # LHamiltonian and RsuperOP are sparse matrix 
                rho_t[0] = rho          
                for i in range(Npoints-1):
                    rho = U.dot(rho - rhoeq) + rhoeq
                    rho_t[i+1] = rho

            if Pmethod == "Relaxation Lindblad":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho)
                    rho_t[i+1] = rho 

            if Pmethod == "Relaxation ShapedPulse Lindblad_":
                rho_t = np.zeros(
                    (Npoints, self.Ldim, 1),
                    dtype=complex
                )

                rho_t[0] = rho

                for i in range(Npoints - 1):

                    # Evaluate the time-dependent Hamiltonian
                    # at one scalar time point
                    H_shapePulse = self.TimeDependent_Hamiltonian(t[i])

                    U = expm(
                        -1j * (Hamiltonian + H_shapePulse) * dt
                        - Relaxation * dt
                    )

                    rho = U @ rho
                    rho_t[i + 1] = rho

            if Pmethod == "Relaxation ShapedPulse Lindblad":

                rho_t = np.zeros(
                    (Npoints, self.Ldim, 1),
                    dtype=complex
                )

                rho_t[0] = rho

                for i in range(Npoints - 1):

                    # Evaluate the time-dependent Hamiltonian
                    H_shapePulse = self.TimeDependent_Hamiltonian(t[i])

                    # Generator for one time step
                    A = (
                        -1j * (Hamiltonian + H_shapePulse)
                        - Relaxation
                    ) * dt

                    # Directly calculate exp(A) @ rho
                    rho = expm_multiply(A, rho)

                    rho_t[i + 1] = rho

            if Pmethod == "Relaxation Lindblad Sparse":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                U = sparse.linalg.expm(-1j * Hamiltonian * dt - Relaxation * dt) # LHamiltonian and RsuperOP are sparse matrix
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho)
                    rho_t[i+1] = rho 

            if Pmethod == "ODE Solver":
                """
                Reference: Equation 47, A liouville space formulation of wangsness-bloch-redfield theory of nuclear spin relaxation suitable for machine computation. I. fundamental aspects, Slawomir Szymanski et.al., https://doi.org/10.1016/0022-2364(86)90334-3
                """
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex) 
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                Lrho = np.reshape(rho,rho.shape[0]) + 0 * 1j            
                Lrhoeq = np.reshape(rhoeq,rhoeq.shape[0])
                
                def rhoDOT(t,Lrho,LHamiltonian,RsuperOP,Lrhoeq,Sx,Sy):
                    Brd = self.class_NonL.Radiation_Damping(self.Convert_LrhoTO2Drho(Lrho))
                    LH = LHamiltonian + self.CommutationSuperoperator(np.sum(Sx,axis=0) * Brd.real)  + self.CommutationSuperoperator(np.sum(Sy,axis=0) * Brd.imag)
                    rhodot = np.zeros((self.Ldim),dtype=complex)
                    rhodot = -1j * np.matmul(LH,Lrho) - np.matmul(RsuperOP,Lrho-Lrhoeq)
                    rhodot = np.reshape(rhodot,rhodot.shape[0])
                    return rhodot
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],Lrho,method=ode_method,t_eval=t,args=(Hamiltonian,Relaxation,Lrhoeq,Sx,Sy), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, rho_sol = rhoSol.t, rhoSol.y
                print(rho_sol.shape)
                for i in range(Npoints):
                    rho_t[i] = np.reshape(rho_sol[:,i],(self.Ldim,1))

            if Pmethod == "ODE Solver Lindblad":
                """
                Reference: Equation 47, A liouville space formulation of wangsness-bloch-redfield theory of nuclear spin relaxation suitable for machine computation. I. fundamental aspects, Slawomir Szymanski et.al., https://doi.org/10.1016/0022-2364(86)90334-3
                """
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex) 
                #t = np.arange(Npoints) * dt # Vineeth
                #t = np.linspace(0,dt*Npoints,Npoints,endpoint=True) # Vineeth
                #t = np.linspace(0,dt*(Npoints-1),Npoints,endpoint=True) # John Price

                Lrho = np.reshape(rho,rho.shape[0]) + 0 * 1j            
                Lrhoeq = np.reshape(rhoeq,rhoeq.shape[0])
                
                def rhoDOT(t,Lrho,LHamiltonian,RsuperOP,Sx,Sy):
                    Brd = self.class_NonL.Radiation_Damping(self.Convert_LrhoTO2Drho(Lrho))
                    LH = LHamiltonian + self.CommutationSuperoperator(np.sum(Sx,axis=0) * Brd.real)  + self.CommutationSuperoperator(np.sum(Sy,axis=0) * Brd.imag)
                    rhodot = np.zeros((self.Ldim),dtype=complex)
                    rhodot = -1j * np.matmul(LH,Lrho) - np.matmul(RsuperOP,Lrho)
                    rhodot = np.reshape(rhodot,rhodot.shape[0])
                    return rhodot
                rhoSol = solve_ivp(rhoDOT,[0,dt*(Npoints-1)],Lrho,method=ode_method,t_eval=t,args=(Hamiltonian,Relaxation,Sx,Sy), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, rho_sol = rhoSol.t, rhoSol.y
                print(rho_sol.shape)
                for i in range(Npoints):
                    rho_t[i] = np.reshape(rho_sol[:,i],(self.Ldim,1))

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