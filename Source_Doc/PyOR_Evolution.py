"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain class Evolutions
"""

import numpy as np
from numpy import linalg as lina
import re
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy import sparse

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
    def __init__(self, class_QS,class_Ham):
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
        self.Npoints = int(self.AcqAQ/self.AcqDT)
        self.ShapeParOmega = self.class_QS.ShapeParOmega
        self.ShapeParFreq = self.class_QS.ShapeParFreq
        self.ShapeParPhase = self.class_QS.ShapeParPhase
        self.Vdim = self.class_QS.Vdim
        self.Ldim = self.class_QS.Ldim
        self.ODE_atol = self.class_QS.ODE_atol
        self.ODE_rtol = self.class_QS.ODE_rtol
        self.ShapeFunc = self.class_QS.ShapeFunc
        self.Maser_TempGradient = self.class_QS.Maser_TempGradient
        self.Lindblad_FinalInverseTemp = self.class_QS.Lindblad_FinalInverseTemp
        self.Lindblad_InitialInverseTemp = self.class_QS.Lindblad_InitialInverseTemp
        self.Lindblad_Temp = self.class_QS.Lindblad_Temp


    def Update(self):
        self.PropagationSpace = self.class_QS.PropagationSpace
        self.PropagationMethod = self.class_QS.PropagationMethod
        self.OdeMethod = self.class_QS.OdeMethod
        self.AcqAQ = self.class_QS.AcqAQ
        self.AcqDT = self.class_QS.AcqDT
        self.Npoints = int(self.AcqAQ/self.AcqDT)
        self.ShapeParOmega = self.class_QS.ShapeParOmega
        self.ShapeParFreq = self.class_QS.ShapeParFreq
        self.ShapeParPhase = self.class_QS.ShapeParPhase
        self.Vdim = self.class_QS.Vdim
        self.Ldim = self.class_QS.Ldim
        self.ODE_atol = self.class_QS.ODE_atol
        self.ODE_rtol = self.class_QS.ODE_rtol
        self.ShapeFunc = self.class_QS.ShapeFunc
        self.Maser_TempGradient = self.class_QS.Maser_TempGradient
        self.Lindblad_FinalInverseTemp = self.class_QS.Lindblad_FinalInverseTemp
        self.Lindblad_Temp = self.class_QS.Lindblad_Temp

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Time evolution of Density Matrix in Hilbert Space
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

        
    def TimeDependent_Hamiltonian(self,t):
        """
        """
        
        if self.ShapeFunc == "Off Resonance":
            return self.class_Ham.Zeeman_B1_Offresonance(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)
        if self.ShapeFunc == "Bruker":
            return self.class_Ham.Zeeman_B1_ShapedPulse(t,self.ShapeParOmega,-1*self.ShapeParFreq,self.ShapeParPhase)        

    def TimeDependent_Hamiltonian_Hilbert(self,t):
        """
        """
        H_shape = np.zeros((t.shape[-1],self.Vdim,self.Vdim),dtype=np.double)
        for i in range(t.shape[-1]):
            H_shape[i] = self.TimeDependent_Hamiltonian(t[i]).real
        return H_shape   

    def Evolution(self,rhoQ,rhoeqQ,HamiltonianQ,RelaxationQ=None,HamiltonianArray=None):

        Pmethod = self.PropagationMethod
        ode_method = self.OdeMethod
        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

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
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt)
                
                for i in range(Npoints):
                    vec_ = np.matmul(U,vec_)
                    vec_t.append(vec_)

            if Pmethod == "ODE Solver":
                vec_ = rhoQ.data
                vec_t = []

                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)

                Lvec = vec_.flatten().astype(complex)  # Ensure it's a 1D complex array
                t = np.linspace(0, dt * Npoints, Npoints, endpoint=True)

                def vecDOT(t, Lvec, Hamiltonian):
                    return -1j * Hamiltonian @ Lvec  # No need for redundant reshaping
            
                vecSol = solve_ivp(vecDOT,[0,dt*Npoints],Lvec,method=self.OdeMethod,t_eval=t,args=(Hamiltonian,), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, vec_sol = vecSol.t, vecSol.y

                for i in range(Npoints):
                    vec_t.append(np.reshape(vec_sol[:,i],(vec_.shape[0],1)))

            return t, vec_t


        if self.PropagationSpace == "Hilbert":
            
            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,np.matmul(rho,U.T.conj()))
                    rho_t[i+1] = rho   

            if Pmethod == "Unitary Propagator Time Dependent":    
                rho_t = np.zeros((Npoints,self.Vdim,self.Vdim),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
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
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
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
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=ode_method,t_eval=t,args=(rhoeq,Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
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
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                rhoi = rho.reshape(-1) + 0 * 1j
                def rhoDOT(t,rho,Hamiltonian,Sx,Sy,Sz,Sp,Sm):                    
                    rho_temp = np.reshape(rho,(self.Vdim,self.Vdim))
                    rhodot = np.zeros((rhoi.shape[-1]))
                    if self.Maser_TempGradient:
                        TempTemp = round(self.class_Relax.Lindblad_TemperatureGradient(t),6)
                        if self.Lindblad_InitialInverseTemp < 0:
                            if TempTemp <= self.Lindblad_FinalInverseTemp:
                                self.class_QS.Lindblad_Temp = TempTemp
                            else:
                                self.class_QS.Lindblad_Temp = self.Lindblad_FinalInverseTemp
                        else:
                            if TempTemp >= self.Lindblad_FinalInverseTemp:
                                self.class_QS.Lindblad_Temp = TempTemp
                            else:
                                self.class_QS.Lindblad_Temp = self.Lindblad_FinalInverseTemp
                        print(f"\rt = {t:0.3f}  Temp = {TempTemp:0.3f}  Lindblad_Temp = {self.class_QS.Lindblad_Temp:0.3f}", end='')                        
                    Rso_temp = self.class_Relax.Relaxation(rho_temp)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
                    rhodot = (-1j * self.Commutator(H,rho_temp) - Rso_temp).reshape(-1)        
                    return rhodot  
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],rhoi,method=ode_method,t_eval=t,args=(Hamiltonian,Sx,Sy,Sz,Sp,Sm), atol = self.ODE_atol, rtol = self.ODE_rtol)
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
                    Rso_temp = self.class_Relax.Relaxation(rho_temp-rhoeq)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    #Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H_shapePulse = self.TimeDependent_Hamiltonian(t)
                    #H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole + H_shapePulse 
                    H = H_shapePulse + Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag
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
                    Rprocess2 = "Phenomenological"
                    Rso_temp = self.class_Relax.Relaxation(rho_temp-rhoeq) + self.class_Relax.Relaxation(rho_temp-rhoeq,Rprocess2)
                    Brd = self.class_NonL.Radiation_Damping(rho_temp)
                    Bdipole = self.class_NonL.DipoleShift(rho_temp)
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag  + np.sum(Sz,axis=0) * Bdipole     
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
                    Rso = self.class_Relax.Relaxation(rho-rhoeq)
                    Brd = self.class_NonL.Radiation_Damping(rho)                 
                    H = Hamiltonian + np.sum(Sx,axis=0) * Brd.real + np.sum(Sy,axis=0) * Brd.imag        
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

            if Pmethod == "Unitary Propagator":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho)  
                    rho_t[i+1] = rho  

            if Pmethod == "Unitary Propagator Sparse":  
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = sparse.linalg.expm(-1j * Hamiltonian * dt) # LHamiltonian is sparse matrix
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = U.dot(rho)  
                    rho_t[i+1] = rho
            
            if Pmethod == "Relaxation":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho - rhoeq) + rhoeq
                    rho_t[i+1] = rho        

            if Pmethod == "Relaxation Sparse":   
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = sparse.linalg.expm(-1j * Hamiltonian * dt - Relaxation * dt) # LHamiltonian and RsuperOP are sparse matrix 
                rho_t[0] = rho          
                for i in range(Npoints-1):
                    rho = U.dot(rho - rhoeq) + rhoeq
                    rho_t[i+1] = rho

            if Pmethod == "Relaxation Lindblad":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                U = expm(-1j * Hamiltonian * dt - Relaxation * dt)
                rho_t[0] = rho
                for i in range(Npoints-1):
                    rho = np.matmul(U,rho)
                    rho_t[i+1] = rho 

            if Pmethod == "Relaxation Lindblad Sparse":    
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex)
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
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
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                Lrho = np.reshape(rho,rho.shape[0]) + 0 * 1j            
                Lrhoeq = np.reshape(rhoeq,rhoeq.shape[0])
                
                def rhoDOT(t,Lrho,LHamiltonian,RsuperOP,Lrhoeq,Sx,Sy):
                    Brd = self.class_NonL.Radiation_Damping(self.Convert_LrhoTO2Drho(Lrho))
                    LH = LHamiltonian + self.CommutationSuperoperator(np.sum(Sx,axis=0) * Brd.real)  + self.CommutationSuperoperator(np.sum(Sy,axis=0) * Brd.imag)
                    rhodot = np.zeros((self.Ldim),dtype=complex)
                    rhodot = -1j * np.matmul(LH,Lrho) - np.matmul(RsuperOP,Lrho-Lrhoeq)
                    rhodot = np.reshape(rhodot,rhodot.shape[0])
                    return rhodot
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],Lrho,method=ode_method,t_eval=t,args=(Hamiltonian,Relaxation,Lrhoeq,Sx,Sy), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, rho_sol = rhoSol.t, rhoSol.y
                print(rho_sol.shape)
                for i in range(Npoints):
                    rho_t[i] = np.reshape(rho_sol[:,i],(self.Ldim,1))

            if Pmethod == "ODE Solver Lindblad":
                """
                Reference: Equation 47, A liouville space formulation of wangsness-bloch-redfield theory of nuclear spin relaxation suitable for machine computation. I. fundamental aspects, Slawomir Szymanski et.al., https://doi.org/10.1016/0022-2364(86)90334-3
                """
                rho_t = np.zeros((Npoints,self.Ldim,1),dtype=complex) 
                t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
                Lrho = np.reshape(rho,rho.shape[0]) + 0 * 1j            
                Lrhoeq = np.reshape(rhoeq,rhoeq.shape[0])
                
                def rhoDOT(t,Lrho,LHamiltonian,RsuperOP,Sx,Sy):
                    Brd = self.class_NonL.Radiation_Damping(self.Convert_LrhoTO2Drho(Lrho))
                    LH = LHamiltonian + self.CommutationSuperoperator(np.sum(Sx,axis=0) * Brd.real)  + self.CommutationSuperoperator(np.sum(Sy,axis=0) * Brd.imag)
                    rhodot = np.zeros((self.Ldim),dtype=complex)
                    rhodot = -1j * np.matmul(LH,Lrho) - np.matmul(RsuperOP,Lrho)
                    rhodot = np.reshape(rhodot,rhodot.shape[0])
                    return rhodot
                rhoSol = solve_ivp(rhoDOT,[0,dt*Npoints],Lrho,method=ode_method,t_eval=t,args=(Hamiltonian,Relaxation,Sx,Sy), atol = self.ODE_atol, rtol = self.ODE_rtol)   
                t, rho_sol = rhoSol.t, rhoSol.y
                print(rho_sol.shape)
                for i in range(Npoints):
                    rho_t[i] = np.reshape(rho_sol[:,i],(self.Ldim,1))

            return t, rho_t 
            
    def Expectation(self,rho_t,detectionQ):

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)
        detection = detectionQ.data

        if self.PropagationSpace == "Schrodinger":
            t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
            signal = np.zeros(Npoints,dtype=complex)
            for i in range(Npoints):
                signal[i] = np.trace(np.matmul(rho_t[i].conj().T,np.matmul(detection,rho_t[i]))) 
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
                signal[i] = np.trace(detection.T @ rho_t[i])
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