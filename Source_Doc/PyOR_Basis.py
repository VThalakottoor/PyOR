"""
PyOR Python On Resonance

Author: Vineeth Francis Thalakottoor Jose Chacko

Email: vineethfrancis.physics@gmail.com

Description:
    This file contains the class `Basis`.
"""

import numpy as np
from numpy import linalg as lina
import re
from IPython.display import display, Latex, Math
from sympy.physics.quantum.cg import CG
from fractions import Fraction

try:
    from .PyOR_QuantumObject import QunObj  # For Sphinx and package usage
except ImportError:
    from PyOR_QuantumObject import QunObj   # For scripts or notebooks


class Basis:    
    def __init__(self, class_QS):
        """
        Initialize the Basis class with a quantum system.

        Parameters
        ----------
        class_QS : object
            An instance of a quantum system (expected to contain spin information and operators).
        """
        self.class_QS = class_QS
        self.Return_KetState_Component = False

    def BasisChange_TransformationMatrix(self, old, new):
        r"""
        Compute the transformation matrix between two basis sets.

        The transformation matrix :math:`U` satisfies:
        :math:`| \text{new} \rangle = U | \text{old} \rangle` and :math:`O_{\text{new}} = U O_{\text{old}} U^\dagger`

        Parameters
        ----------
        old : list of QunObj
            Old basis vectors.
        new : list of QunObj
            New basis vectors.

        Returns
        -------
        QunObj
            Transformation matrix as a QunObj.
        """

        if not (isinstance(old, list) and isinstance(new, list)):
            raise TypeError("Both inputs must be lists.")

        if not all(isinstance(item, QunObj) for item in old) or not all(isinstance(item, QunObj) for item in new):
            raise TypeError("All elements in both lists must be instances of QunObj.")

        dim = len(old)
        U = np.zeros((dim,dim),dtype=np.cdouble)
        for i in range(dim):
            for j in range(dim):
                U[i][j] = self.Adjoint((old[i].data)) @ (new[j].data)
        return QunObj(U) 

    def BasisChange_HamiltonianEigenStates(self, H):
        """
        Transform the basis to the eigenbasis of a given Hamiltonian.

        This method computes the eigenvectors of the input Hamiltonian `H` and
        expresses them in the Zeeman basis. It stores the resulting transformation
        matrix in the quantum system, updates the operator basis, and returns
        the eigenvectors and their symbolic representations.

        Parameters
        ----------
        H : QunObj or np.ndarray
            The Hamiltonian for which the eigenbasis should be computed.

        Returns
        -------
        eigenvectors : np.ndarray
            Eigenvectors of the Hamiltonian.
        Dic : list of str
            Symbolic expressions of the eigenvectors in the Zeeman basis.
        """
        Dic = []

        # Diagonalize the Hamiltonian
        eigenvalues, eigenvectors = self.class_QS.Class_quantumlibrary.Eigen_Split(H)

        # Get Zeeman basis states and their labels
        Zstates, DicZ = self.Zeeman_Basis()

        self.Return_KetState_Component = True
        # Convert each eigenvector to a readable label
        for i in eigenvectors:
            label = self.KetState_Components(Zstates, DicZ, i)

            if isinstance(label, str) and label.startswith("Ket State ="):
                label = label[len("Ket State = "):].strip()

            Dic.append(label if label else "undefined")

        self.Return_KetState_Component = False

        # Transformation matrix
        U = self.BasisChange_TransformationMatrix(Zstates, eigenvectors)

        # Store and update system basis
        self.class_QS.Basis_SpinOperators_TransformationMatrix = U
        self.class_QS.Basis_SpinOperators_Hilbert = "Hamiltonian eigen states"
        self.class_QS.Update()

        return eigenvectors, Dic

    def BasisChange_State(self, state, U):
        """
        Transform a state vector using the given transformation matrix.

        Parameters
        ----------
        state : QunObj
            State in the original basis.
        U : QunObj
            Transformation matrix.

        Returns
        -------
        QunObj
            State in the new basis.
        """
        if not isinstance(state, QunObj) or not isinstance(U, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        return QunObj(U.data @ state.data)

    def BasisChange_States(self, states, U):
        """
        Transform one or more state vectors using the given transformation matrix.

        Parameters
        ----------
        states : QunObj or list of QunObj
            State(s) in the original basis.
        U : QunObj
            Transformation matrix.

        Returns
        -------
        QunObj or list of QunObj
            State(s) in the new basis.
        """
        if not isinstance(U, QunObj):
            raise TypeError("U must be an instance of QunObj.")
        
        # Handle a single state
        if isinstance(states, QunObj):
            return QunObj(U.data @ states.data)
        
        # Handle a list of states
        if isinstance(states, list):
            if not all(isinstance(s, QunObj) for s in states):
                raise TypeError("All elements in the list must be instances of QunObj.")
            return [QunObj(U.data @ s.data) for s in states]

        raise TypeError("states must be a QunObj or a list of QunObj.")

    def BasisChange_Operator(self, O, U):
        """
        Transform an operator using the given transformation matrix.

        Parameters
        ----------
        O : QunObj
            Operator in the original basis.
        U : QunObj
            Transformation matrix.

        Returns
        -------
        QunObj
            Operator in the new basis.
        """
        if not isinstance(O, QunObj) or not isinstance(U, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        return QunObj(self.Adjoint(U.data) @ O.data @ U.data) 

    def BasisChange_SpinOperators(self, Sop, U):
        """
        Transform a list of spin operators using a transformation matrix.

        Parameters
        ----------
        Sop : list of QunObj
            Spin operators in the original basis.
        U : QunObj
            Transformation matrix.

        Returns
        -------
        list of QunObj
            Transformed spin operators.
        """
        if not isinstance(Sop, list):
            raise TypeError("input must be lists.")

        if not all(isinstance(item, QunObj) for item in Sop):
            raise TypeError("All elements in the list must be instances of QunObj.")

        if not isinstance(U, QunObj):
            raise TypeError("Input must be instances of QunObj.")

        Sop_N = []
        for i in range(len(Sop)):
            transformed = QunObj(U.data @ Sop[i].data @ self.Adjoint(U.data))
            Sop_N.append(transformed)
        return Sop_N


    def KetState_Components(self, AQ, dic, ketQ):
        """
        Decompose ket state into basis components.

        Parameters
        ----------
        AQ : list of QunObj
            Basis ket vectors (assumed orthonormal).
        dic : dict
            Dictionary mapping basis indices to basis labels.
        ketQ : QunObj
            Ket state vector to be decomposed.

        Returns
        -------
        None
        """
        if not isinstance(AQ, list):
            raise TypeError("Input must be a list.")
        if not all(isinstance(item, QunObj) for item in AQ):
            raise TypeError("All elements must be instances of QunObj.")

        psi = ketQ.data
        if psi.shape[1] != 1:
            raise ValueError("Input state must be a column vector (ket).")

        components = np.array([self.InnerProduct(A.data, psi) for A in AQ])
        tol = 1.0e-10
        components.real[abs(components.real) < tol] = 0.0
        components.imag[abs(components.imag) < tol] = 0.0

        output = ["Ket State = "]
        for i, val in enumerate(components):
            if val != 0:
                comp_str = f"{round(val.real, 5)}" if val.imag == 0 else \
                        f"{round(val.real, 5)} + {round(val.imag, 5)}j" if val.real != 0 else \
                        f"{round(val.imag, 5)}j"
                output.append(f"{comp_str} {dic[i]} + ")

        print((''.join(output))[:-3])

        if self.Return_KetState_Component:
            return ''.join(output)[:-3]

    def CG_Coefficient(self, j1, m1, j2, m2, J, M):
        """
        Compute the Clebsch-Gordan coefficient ⟨j1 m1 j2 m2 | J M⟩.

        Parameters
        ----------
        j1, m1, j2, m2, J, M : float or int
            Quantum numbers for the Clebsch-Gordan coefficient.

        Returns
        -------
        float
            Value of the Clebsch-Gordan coefficient.
        """
        return float(CG(j1, m1, j2, m2, J, M).doit())

    def Spherical_OpBasis(self, S):
        """
        Generate spherical tensor operator basis for a single spin.

        Parameters
        ----------
        S : float
            Spin quantum number.

        Returns
        -------
        list of QunObj
            Spherical tensor operators.
        list of int
            Corresponding coherence orders.
        list of tuple
            List of (L, M) values.

        Reference:
        ----------
        1. Quantum Theory of Angular Momentum, D. A. Varshalovich, A. N. Moskalev and V. K. Khersonskii
        """
        states = int(2 * S + 1)
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
                Pol_Basis.append(QunObj(np.sqrt((2*i + 1)/(2*S+1)) * Sum))
                Coherence_order.append(j) 
                LM_state.append(tuple([i,j]))
        
        return Pol_Basis,Coherence_order,LM_state                 

    def ProductOperators_SphericalTensor(self, sort='negative to positive', Index=False):
        """
        Generate spherical tensor basis for a multi-spin system.

        Parameters
        ----------
        sort : str, optional
            Sorting option for coherence order ('normal', 'negative to positive', 'zero to high').
        Index : bool, optional
            Whether to append index to labels.

        Returns
        -------
        list of QunObj
            List of spherical tensor operators for the multi-spin system.
        list of int
            Corresponding coherence orders.
        list of str
            Labels of each basis operator in the form T(L,M).
        """
        spin_list = self.class_QS.slist.tolist()

        OP, CO, LM = self.Spherical_OpBasis(spin_list[0])
        DIC = [f"T({L},{M})" for (L, M) in LM]

        for idx in range(1, len(spin_list)):
            OP_next, CO_next, LM_next = self.Spherical_OpBasis(spin_list[idx])
            DIC_next = [f"T({L},{M})" for (L, M) in LM_next]

            OP, CO, DIC = self.ProductOperator(OP, CO, DIC, OP_next, CO_next, DIC_next, sort=sort, indexing=Index)

        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return OP, CO, DIC
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(OP,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), CO, DIC    
                    
        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(OP), CO, DIC        

    def ProductOperators_SphericalTensor_Test(self, sort='negative to positive', Index=False):
        """
        Generate spherical tensor basis for a multi-spin system.

        Parameters
        ----------
        sort : str, optional
            Sorting option for coherence order ('normal', 'negative to positive', 'zero to high').
        Index : bool, optional
            Whether to append index to labels.

        Returns
        -------
        list of QunObj
            List of spherical tensor operators for the multi-spin system.
        list of int
            Corresponding coherence orders.
        list of str
            Labels of each basis operator in the form SpinLabel(L,M).
        """
        spin_list = self.class_QS.slist.tolist()
        SpinLabels = self.class_QS.SpinDic

        # First spin
        OP, CO, LM = self.Spherical_OpBasis(spin_list[0])
        DIC = [f"{SpinLabels[0]}({L},{M})" for (L, M) in LM]

        # Loop over remaining spins
        for idx in range(1, len(spin_list)):
            OP_next, CO_next, LM_next = self.Spherical_OpBasis(spin_list[idx])
            DIC_next = [f"{SpinLabels[idx]}({L},{M})" for (L, M) in LM_next]

            OP, CO, DIC = self.ProductOperator(
                OP, CO, DIC,
                OP_next, CO_next, DIC_next,
                sort=sort, indexing=Index
            )

        # Return based on propagation space
        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return OP, CO, DIC
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(
                    OP,
                    self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()
                ), CO, DIC

        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(OP), CO, DIC

    def ProductOperator(self, OP1, CO1, DIC1, OP2, CO2, DIC2, sort, indexing):
        """
        Perform the Kronecker product of two sets of spherical basis operators.

        Parameters
        ----------
        OP1, OP2 : list of QunObj
            Basis operators of each subsystem.
        CO1, CO2 : list of int
            Coherence orders for each subsystem.
        DIC1, DIC2 : list of str
            Labels for each subsystem.
        sort : str
            Sorting method for coherence order.
        indexing : bool
            Whether to append indices to the labels.

        Returns
        -------
        list of QunObj
            Combined operator basis.
        list of int
            Combined coherence orders.
        list of str
            Combined operator labels.
        """
        if not (isinstance(OP1, list) and isinstance(OP2, list)):
            raise TypeError("Both inputs must be lists.")

        if not all(isinstance(item, QunObj) for item in OP1) or not all(isinstance(item, QunObj) for item in OP2):
            raise TypeError("All elements in both lists must be instances of QunObj.")

        CO = []
        OP = []
        DIC = []
        index = 0
        for i,j,k in zip(OP1,CO1,DIC1):
            for m,n,o in zip(OP2,CO2,DIC2):
                OP.append(QunObj(np.kron(i.data,m.data)))
                CO.append(j+n)
                DIC.append(k+o)
                
        if sort == 'negative to positive':        
            combine = list(zip(CO,OP,DIC))
            combine_sort = sorted(combine, key=lambda x: x[0])
            Sort_CO,Sort_OP,Sort_DIC = zip(*combine_sort)  
            CO = list(Sort_CO)
            OP = list(Sort_OP)
            DIC = list(Sort_DIC)      
            
        if sort == 'zero to high':        
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

    def ProductOperators_SpinHalf_Cartesian(self, Index=False, Normal=True):
        """
        Generate product operator basis in the Cartesian basis for spin-1/2 systems.

        Parameters
        ----------
        Index : bool, optional
            Whether to include index in labels.
        Normal : bool, optional
            Whether to normalize the operators.

        Returns
        -------
        list of QunObj
            Product operators.
        list of str
            Corresponding labels.
        """ 
        Dic = ["Id ","Ix ","Iy ","Iz "]
        Single_OP = self.class_QS.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [QunObj(np.eye(2)),QunObj(Single_OP[0]),QunObj(Single_OP[1]),QunObj(Single_OP[2])]
        
        Coherence_order_SpinHalf = list(range(len(Dic)))
        
        Basis_SpinHalf_out = []
        Dic_out = []
        Coherence_order_SpinHalf_out = []
                
        if self.class_QS.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Dic_out = Dic
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            sort = 'normal'
            for i in range(self.class_QS.Nspins-1):
                if i == self.class_QS.Nspins-2:
                    indexing = Index
                if i == 0:    
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic] 
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]     
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                    Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                    Basis_SpinHalf, Coherence_order_SpinHalf, Dic, sort, indexing)                
        
        if Normal:
            for j in range(self.class_QS.Ldim):
                Basis_SpinHalf_out[j] = QunObj(self.Normalize(Basis_SpinHalf_out[j].data))
        
        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return Basis_SpinHalf_out, Dic_out 
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(Basis_SpinHalf_out,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), Dic_out 
                        
        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Dic_out

    def ProductOperators_SpinHalf_Cartesian_Test(self, Index=False, Normal=True):
        """
        Generate product operator basis in the Cartesian basis for spin-1/2 systems.

        Parameters
        ----------
        Index : bool, optional
            Whether to include index in labels.
        Normal : bool, optional
            Whether to normalize the operators.

        Returns
        -------
        list of QunObj
            Product operators.
        list of str
            Corresponding labels.
        """ 
        Dic_base = ["id", "x", "y", "z"]
        SpinLabels = self.class_QS.SpinDic

        Single_OP = self.class_QS.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [
            QunObj(np.eye(2)),
            QunObj(Single_OP[0]),
            QunObj(Single_OP[1]),
            QunObj(Single_OP[2])
        ]

        Dic = [f"{SpinLabels[0]}{op}" for op in Dic_base]
        Coherence_order_SpinHalf = list(range(len(Dic_base)))

        Basis_SpinHalf_out = Basis_SpinHalf
        Dic_out = Dic
        Coherence_order_SpinHalf_out = Coherence_order_SpinHalf

        for i in range(1, self.class_QS.Nspins):
            Dic_next = [f"{SpinLabels[i]}{op}" for op in Dic_base]
            indexing = Index if i == self.class_QS.Nspins - 1 else False
            Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                Basis_SpinHalf, Coherence_order_SpinHalf, Dic_next, sort='normal', indexing=indexing
            )

        # Clean Dic_out: robust spin-op parsing and cleanup
        cleaned_Dic_out = []
        cleaned_Basis_out = []

        valid_suffixes = {"id", "x", "y", "z"}
        spin_labels = self.class_QS.SpinDic

        for label, op in zip(Dic_out, Basis_SpinHalf_out):
            i = 0
            parsed_terms = []

            while i < len(label):
                matched = False
                for spin_label in spin_labels:
                    if label.startswith(spin_label, i):
                        for suffix in valid_suffixes:
                            full_term = spin_label + suffix
                            if label.startswith(full_term, i):
                                parsed_terms.append(full_term)
                                i += len(full_term)
                                matched = True
                                break
                        if matched:
                            break
                if not matched:
                    i += 1  # Skip unrecognized junk (should not happen)

            # Remove 'id' terms
            reduced_terms = [term for term in parsed_terms if not term.endswith("id")]

            if reduced_terms:
                new_label = ''.join(reduced_terms)
            else:
                new_label = "id"

            cleaned_Dic_out.append(new_label)
            cleaned_Basis_out.append(op)

        Dic_out = cleaned_Dic_out
        Basis_SpinHalf_out = cleaned_Basis_out

        if Normal:
            for j in range(len(Basis_SpinHalf_out)):
                Basis_SpinHalf_out[j] = QunObj(self.Normalize(Basis_SpinHalf_out[j].data))

        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return Basis_SpinHalf_out, Dic_out 
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(
                    Basis_SpinHalf_out,
                    self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()
                ), Dic_out

        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Dic_out

    def ProductOperators_SpinHalf_PMZ(self, sort='negative to positive', Index=False, Normal=True):
        """
        Generate product operators for spin-1/2 systems in the PMZ basis.

        Parameters
        ----------
        sort : str, optional
            Sorting method for coherence order.
        Index : bool, optional
            Whether to include index in labels.
        Normal : bool, optional
            Whether to normalize the operators.

        Returns
        -------
        list of QunObj
            Product operators.
        list of int
            Coherence orders.
        list of str
            Operator labels.
        """ 
        Dic = ["Im ","Iz ","Id ","Ip "]
        Single_OP = self.class_QS.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [
            QunObj(Single_OP[0] - 1j * Single_OP[1]),
            QunObj(Single_OP[2]),
            QunObj(np.eye(2)),
            QunObj(-1 * (Single_OP[0] + 1j * Single_OP[1]))
        ]
        
        Coherence_order_SpinHalf = [-1, 0, 0, 1]
        
        Basis_SpinHalf_out = []
        Dic_out = []
        Coherence_order_SpinHalf_out = []
                
        if self.class_QS.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Dic_out = Dic
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            for i in range(self.class_QS.Nspins - 1):
                if i == self.class_QS.Nspins - 2:
                    indexing = Index
                if i == 0:
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic]
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                    Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                    Basis_SpinHalf, Coherence_order_SpinHalf, Dic, sort, indexing)

        if Normal:
            for j in range(self.class_QS.Ldim):
                Basis_SpinHalf_out[j] = QunObj(self.Normalize(Basis_SpinHalf_out[j].data))

        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out 
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(Basis_SpinHalf_out,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), Coherence_order_SpinHalf_out, Dic_out 
                        
        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Coherence_order_SpinHalf_out, Dic_out

    def ProductOperators_SpinHalf_PMZ_Test(self, sort='negative to positive', Index=False, Normal=True):
        """
        Generate product operators for spin-1/2 systems in the PMZ basis.

        Parameters
        ----------
        sort : str, optional
            Sorting method for coherence order.
        Index : bool, optional
            Whether to include index in labels.
        Normal : bool, optional
            Whether to normalize the operators.

        Returns
        -------
        list of QunObj
            Product operators.
        list of int
            Coherence orders.
        list of str
            Operator labels.
        """ 
        Dic_base = ["m", "z", "id", "p"]
        Coherence_order_SpinHalf = [-1, 0, 0, 1]
        SpinLabels = self.class_QS.SpinDic

        # Single-spin PMZ operators
        Single_OP = self.class_QS.SpinOperatorsSingleSpin(1/2).astype(np.complex64)
        Basis_SpinHalf = [
            QunObj(Single_OP[0] - 1j * Single_OP[1]),              # m
            QunObj(Single_OP[2]),                                  # z
            QunObj(np.eye(2)),                                     # id
            QunObj(-1 * (Single_OP[0] + 1j * Single_OP[1]))        # p
        ]

        # Initial labels
        Dic = [f"{SpinLabels[0]}{op}" for op in Dic_base]

        # Initialize outputs
        Basis_SpinHalf_out = Basis_SpinHalf
        Dic_out = Dic
        Coherence_order_SpinHalf_out = Coherence_order_SpinHalf

        # Loop for multiple spins
        for i in range(1, self.class_QS.Nspins):
            Dic_next = [f"{SpinLabels[i]}{op}" for op in Dic_base]
            indexing = Index if i == self.class_QS.Nspins - 1 else False
            Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                Basis_SpinHalf, Coherence_order_SpinHalf, Dic_next, sort, indexing
            )

        # Clean Dic_out: remove spin terms ending in 'id'
        cleaned_Dic_out = []
        cleaned_Basis_out = []

        valid_suffixes = {"m", "z", "id", "p"}

        for label, op in zip(Dic_out, Basis_SpinHalf_out):
            i = 0
            parsed_terms = []
            while i < len(label):
                matched = False
                for spin_label in SpinLabels:
                    for suffix in valid_suffixes:
                        full_term = spin_label + suffix
                        if label.startswith(full_term, i):
                            parsed_terms.append(full_term)
                            i += len(full_term)
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    i += 1  # Skip unrecognized part

            # Remove terms ending in 'id'
            reduced_terms = [term for term in parsed_terms if not term.endswith("id")]
            new_label = ''.join(reduced_terms) if reduced_terms else "id"

            cleaned_Dic_out.append(new_label)
            cleaned_Basis_out.append(op)

        Dic_out = cleaned_Dic_out
        Basis_SpinHalf_out = cleaned_Basis_out

        # Normalize
        if Normal:
            for j in range(len(Basis_SpinHalf_out)):
                Basis_SpinHalf_out[j] = QunObj(self.Normalize(Basis_SpinHalf_out[j].data))

        # Return based on propagation space
        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_SpinOperators(
                    Basis_SpinHalf_out,
                    self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()
                ), Coherence_order_SpinHalf_out, Dic_out

        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Coherence_order_SpinHalf_out, Dic_out

    def ProductOperators_SpinHalf_SphericalTensor(self, sort='negative to positive', Index=False):
        """
        Generate product operators for spin-1/2 systems in the spherical tensor basis.

        Parameters
        ----------
        sort : str, optional
            Sorting method for coherence order.
        Index : bool, optional
            Whether to include index in labels.

        Returns
        -------
        list of QunObj
            Product operators.
        list of int
            Coherence orders.
        list of str
            Operator labels.
        """        
        Dic = ["Id ","Im ","Iz ","Ip "]
        Basis_SpinHalf, Coherence_order_SpinHalf, LM_state_SpinHalf = self.Spherical_OpBasis(1/2)
        Basis_SpinHalf_out = []
        Coherence_order_SpinHalf_out = []
        Dic_out = []
        
        if self.class_QS.Nspins == 1:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
        else:
            Basis_SpinHalf_out = Basis_SpinHalf
            Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
            Dic_out = Dic
            Dic_out = [s.replace(" ", "1 ") for s in Dic_out]
            indexing = False
            
            for i in range(self.class_QS.Nspins-1):    
                if i == self.class_QS.Nspins-2:
                    indexing = Index
                if i == 0:    
                    Dic = [s.replace(" ", str(i+2) + " ") for s in Dic] 
                Dic = [s.replace(str(i+1), str(i+2) + " ") for s in Dic]      
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                    Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                    Basis_SpinHalf, Coherence_order_SpinHalf, Dic, sort, indexing)

        if self.class_QS.PropagationSpace == "Hilbert":   
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":     
                return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out 
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":     
                return self.BasisChange_SpinOperators(Basis_SpinHalf_out,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), Coherence_order_SpinHalf_out, Dic_out 
                        
        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Coherence_order_SpinHalf_out, Dic_out

    def ProductOperators_SpinHalf_SphericalTensor_Test(self, sort='negative to positive', Index=False):
        """
        Generate product operators for spin-1/2 systems in the spherical tensor basis.

        Parameters
        ----------
        sort : str, optional
            Sorting method for coherence order.
        Index : bool, optional
            Whether to include index in labels.

        Returns
        -------
        list of QunObj
            Product operators.
        list of int
            Coherence orders.
        list of str
            Operator labels.
        """
        Dic_base = ["id", "m", "z", "p"]
        SpinLabels = self.class_QS.SpinDic

        # Get single-spin spherical tensor basis
        Basis_SpinHalf, Coherence_order_SpinHalf, LM_state_SpinHalf = self.Spherical_OpBasis(1/2)

        # Build labels for the first spin
        Dic = [f"{SpinLabels[0]}{op}" for op in Dic_base]

        Basis_SpinHalf_out = Basis_SpinHalf
        Coherence_order_SpinHalf_out = Coherence_order_SpinHalf
        Dic_out = Dic

        # Build product operators across spins
        for i in range(1, self.class_QS.Nspins):
            Dic_next = [f"{SpinLabels[i]}{op}" for op in Dic_base]
            indexing = Index if i == self.class_QS.Nspins - 1 else False
            Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out = self.ProductOperator(
                Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out,
                Basis_SpinHalf, Coherence_order_SpinHalf, Dic_next, sort, indexing
            )

        # Clean Dic_out: remove 'id' spin terms
        cleaned_Dic_out = []
        cleaned_Basis_out = []

        valid_suffixes = {"id", "m", "z", "p"}

        for label, op in zip(Dic_out, Basis_SpinHalf_out):
            i = 0
            parsed_terms = []
            while i < len(label):
                matched = False
                for spin_label in SpinLabels:
                    for suffix in valid_suffixes:
                        full_term = spin_label + suffix
                        if label.startswith(full_term, i):
                            parsed_terms.append(full_term)
                            i += len(full_term)
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    i += 1  # Skip unrecognized part

            # Remove 'id' terms only
            reduced_terms = [term for term in parsed_terms if not term.endswith("id")]
            new_label = ''.join(reduced_terms) if reduced_terms else "id"

            cleaned_Dic_out.append(new_label)
            cleaned_Basis_out.append(op)

        Dic_out = cleaned_Dic_out
        Basis_SpinHalf_out = cleaned_Basis_out

        # Return based on propagation space
        if self.class_QS.PropagationSpace == "Hilbert":   
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":     
                return Basis_SpinHalf_out, Coherence_order_SpinHalf_out, Dic_out
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":     
                return self.BasisChange_SpinOperators(
                    Basis_SpinHalf_out,
                    self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()
                ), Coherence_order_SpinHalf_out, Dic_out
                            
        if self.class_QS.PropagationSpace == "Liouville":
            return self.ProductOperators_ConvertToLiouville(Basis_SpinHalf_out), Coherence_order_SpinHalf_out, Dic_out


    def String_to_Matrix(self, dic, Basis):
        """
        Convert a dictionary of labels to a dictionary mapping labels to matrices.

        Parameters
        ----------
        dic : list of str
            Dictionary labels for operator basis.
        Basis : list of QunObj
            Corresponding list of operators.

        Returns
        -------
        dict
            Dictionary mapping cleaned labels to QunObj instances.
        """    
        char_to_remove = "Id"
        dic = [re.sub(f"{re.escape(char_to_remove)}.", " ", s) for s in dic]               
        dic = [s.replace(" ", "") for s in dic]

        print(dic)
        return dict(zip(dic, Basis))

    def ProductOperators_Zeeman(self):
        """
        Generate product operators in the Zeeman basis.

        Returns
        -------
        list of QunObj
            Product operators in Zeeman basis.
        list of str
            Operator labels.
        list of float
            Coherence orders.
        QunObj
            Coherence order as a 2D matrix.
        """
        B_Z, dic_dummy = self.Zeeman_Basis()
        Kets = self.class_QS.ZeemanBasis_Ket()
        Bras = self.class_QS.ZeemanBasis_Bra()
        dic = []
        coh = []
        State_Momentum = self.Basis_Ket_AngularMomentum_Array()
        
        B = []
        for i in range(self.class_QS.Vdim):
            for j in range(self.class_QS.Vdim):
                B.append(QunObj(np.outer(B_Z[i].data, self.Adjoint(B_Z[j].data))))
                dic.append(Kets[i] + Bras[j])
                coh.append(State_Momentum[i] - State_Momentum[j])

        if self.class_QS.PropagationSpace == "Hilbert":
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return B, dic, coh, QunObj(np.asarray(coh).reshape((self.class_QS.Vdim, self.class_QS.Vdim))) 
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return B, dic, coh, QunObj(np.asarray(coh).reshape((self.class_QS.Vdim, self.class_QS.Vdim))) 
                        
        if self.class_QS.PropagationSpace == "Liouville":  
            return self.ProductOperators_ConvertToLiouville(B), dic, coh, self.class_QS.Class_quantumlibrary.DMToVec(QunObj(np.asarray(coh).reshape((self.class_QS.Vdim, self.class_QS.Vdim)))) 
    
    def Zeeman_Basis(self):
        """
        Compute eigenbasis of the total Sz operator (Zeeman basis).

        Returns
        -------
        list of QunObj
            Zeeman basis vectors.
        list of str
            Corresponding basis labels.
        """
        Sz = np.sum(self.class_QS.Sz_, axis=0)
        Dic = self.class_QS.ZeemanBasis_Ket()

        B_Zeeman = []
        eigenvalues, eigenvectors = lina.eig(Sz) 
        for i in range(self.class_QS.Vdim):
            B_Zeeman.append(QunObj((eigenvectors[:, i].reshape(-1, 1)).real))   
        if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":          
            return B_Zeeman, Dic  
        if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
            return self.BasisChange_States(B_Zeeman,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), Dic
        if self.class_QS.Basis_SpinOperators_Hilbert == "Hamiltonian eigen states":
            return self.BasisChange_States(B_Zeeman,self.class_QS.Basis_SpinOperators_TransformationMatrix), Dic
                
    def SingletTriplet_Basis(self): 
        """
        Generate singlet-triplet basis for two spin-1/2 particles.

        Returns
        -------
        list of QunObj
            Singlet-triplet basis vectors.
        list of str
            Basis labels.

        Notes
        -----
        Only works for two spin-1/2 systems.
        """
        Dic = ["S0 ", "Tp ", "T0 ", "Tm "]

        if ((self.class_QS.Nspins == 2) and 
            (self.class_QS.slist[0] == 1/2) and 
            (self.class_QS.slist[1] == 1/2)):
            B_Zeeman, _ = self.Zeeman_Basis()
            B_ST = [
                QunObj(np.array([[0], [1/np.sqrt(2)], [-1/np.sqrt(2)], [0]], dtype=complex)),
                QunObj(np.array([[1], [0], [0], [0]], dtype=complex)),
                QunObj(np.array([[0], [1/np.sqrt(2)], [1/np.sqrt(2)], [0]], dtype=complex)) ,
                QunObj(np.array([[0], [0], [0], [1]], dtype=complex))                
            ]
            if self.class_QS.Basis_SpinOperators_Hilbert == "Zeeman":
                return B_ST, Dic
            if self.class_QS.Basis_SpinOperators_Hilbert == "Singlet Triplet":
                return self.BasisChange_States(B_ST,self.class_QS.Basis_SpinOperators_TransformationMatrix_SingletTriplet.Adjoint()), Dic
        else:
            print("Two spin half system only")

    def Basis_Ket_AngularMomentum_Array(self):
        """
        Compute magnetic quantum numbers for each Zeeman state.

        Returns
        -------
        np.ndarray
            Array of magnetic quantum numbers (diagonal of total Sz).
        """
        Sz = self.class_QS.Sz_
        return (np.sum(Sz, axis=0).real).diagonal()

    def Basis_Ket_AngularMomentum_List(self):
        """
        Compute magnetic quantum numbers for each Zeeman state as strings.

        Returns
        -------
        list of str
            List of magnetic quantum numbers as fractions.
        """
        Sz = self.class_QS.Sz_
        array = (np.sum(Sz, axis=0).real).diagonal()
        List = []
        for i in array:
            List.append(str(Fraction(float(i))))
        return List

    def Normalize(self, A):
        """
        Normalize an operator so its inner product with itself is 1.

        Parameters
        ----------
        A : ndarray
            Operator to normalize.

        Returns
        -------
        ndarray
            Normalized operator.
        """
        return A / np.sqrt(self.InnerProduct(A, A))

    def InnerProduct(self, A, B):
        """
        Compute inner product of two operators.

        Parameters
        ----------
        A : ndarray
        B : ndarray

        Returns
        -------
        complex
            Inner product Tr(A† B)
        """
        return np.trace(np.matmul(A.T.conj(), B))

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

    def ProductOperators_ConvertToLiouville(self, Basis_X):
        """
        Convert product operator basis to Liouville space.

        Parameters
        ----------
        Basis_X : list of QunObj
            Basis in Hilbert space.

        Returns
        -------
        list of QunObj
            Basis in Liouville space.
        """
        dim = len(Basis_X)  
        Basis_out = []
        for i in range(dim): 
            Basis_out.append(QunObj(self.Vector_L(np.asarray(Basis_X[i].data))))
        return Basis_out           

    def Vector_L(self, X):
        """
        Vectorize an operator into Liouville space form.

        Parameters
        ----------
        X : ndarray
            Operator to vectorize.

        Returns
        -------
        ndarray
            Vectorized operator.
        """
        dim = self.class_QS.Vdim
        return np.reshape(X, (dim**2, -1))
