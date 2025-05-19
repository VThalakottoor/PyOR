"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This file contains the class for computing the equilibrium density matrix.

    The equilibrium density matrix is a key concept in magnetic resonance,
    representing the state of the system at thermal equilibrium.
"""


import numpy as np
from scipy.linalg import expm

try:
    from . import PyOR_PhysicalConstants
    from . import PyOR_Rotation
    from .PyOR_QuantumObject import QunObj
    from .PyOR_QuantumLibrary import QuantumLibrary
except ImportError:
    import PyOR_PhysicalConstants
    import PyOR_Rotation
    from PyOR_QuantumObject import QunObj
    from PyOR_QuantumLibrary import QuantumLibrary


QLib = QuantumLibrary()


class DensityMatrix:
    def __init__(self, class_QS, class_HAM):
        self.class_QS = class_QS
        self.class_HAM = class_HAM
        self.hbar = PyOR_PhysicalConstants.constants("hbar")
        self.mu0 = PyOR_PhysicalConstants.constants("mu0")
        self.kb = PyOR_PhysicalConstants.constants("kb")

    def Update(self):
        """Update matrix tolerance from quantum system settings."""

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Equilibrium Density Matrix
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def EquilibriumDensityMatrix(self, Ti, HT_approx=False):
        """
        Calculate equilibrium density matrix for given spin temperatures.

        Parameters
        ----------
        Ti : list or ndarray
            Spin temperatures for each spin.
        HT_approx : bool, optional
            Use high temperature approximation if True.

        Returns
        -------
        QunObj
            Equilibrium density matrix.
        """
        LarmorF = self.class_HAM.LarmorF
        Sz = self.class_QS.Sz_

        H_Eq_T = sum(
            self.hbar * (LarmorF[i] * Sz[i] / (self.kb * Ti[i]))
            for i in range(self.class_QS.Nspins)
        )

        if HT_approx:
            E = np.eye(self.class_QS.Vdim)
            rho_T = (E - H_Eq_T) / np.trace(E - H_Eq_T)
        else:
            rho_T = expm(-H_Eq_T) / np.trace(expm(-H_Eq_T))

        print("Trace of density matrix = ", (np.trace(rho_T)).real)

        if self.class_QS.PropagationSpace == "Hilbert":
            return QunObj(rho_T)

        if self.class_QS.PropagationSpace == "Liouville":
            return self.class_QS.Class_quantumlibrary.DMToVec(QunObj(rho_T)) 
        
    def EquilibriumDensityMatrix_Add_TotalHamiltonian(self, HQ, T, HT_approx=False):
        """
        Calculate equilibrium density matrix for total Hamiltonian.

        Parameters
        ----------
        HQ : QunObj
            Total Hamiltonian.
        T : float
            Uniform spin temperature.
        HT_approx : bool, optional
            Use high temperature approximation if True.

        Returns
        -------
        QunObj
            Equilibrium density matrix.
        """
        H = HQ.data
        H_Eq_T = self.hbar * (H / (self.kb * T))

        if HT_approx:
            E = np.eye(self.class_QS.Vdim)
            rho_T = (E - H_Eq_T) / np.trace(E - H_Eq_T)
        else:
            rho_T = expm(-H_Eq_T) / np.trace(expm(-H_Eq_T))

        print("Trace of density matrix = ", (np.trace(rho_T)).real)

        if self.class_QS.PropagationSpace == "Hilbert":
            return QunObj(rho_T)

        if self.class_QS.PropagationSpace == "Liouville":
            return self.class_QS.Class_quantumlibrary.DMToVec(QunObj(rho_T))

    def InitialDensityMatrix(self, HT_approx=False):
        """Wrapper for equilibrium density matrix using initial temperatures."""
        return self.EquilibriumDensityMatrix(self.class_QS.Ispintemp, HT_approx)

    def FinalDensityMatrix(self, HT_approx=False):
        """Wrapper for equilibrium density matrix using final temperatures."""
        return self.EquilibriumDensityMatrix(self.class_QS.Fspintemp, HT_approx)

    def PolarizationVector(self, spinQ, rhoQ, SzQ, PolPercentage):
        """
        Compute polarization of a spin system.

        Parameters
        ----------
        spinQ : float
            Spin quantum number.
        rhoQ : QunObj
            Density matrix.
        SzQ : QunObj
            Spin-z operator.
        PolPercentage : bool
            Return value as percentage if True.

        Returns
        -------
        float
            Spin polarization value.
        """

        if self.class_QS.PropagationSpace == "Hilbert":
            rho = rhoQ.data
            Sz = SzQ.data
            pol = -(1.0 / spinQ) * np.trace(rho @ Sz).real / np.trace(rho).real
            return 100 * pol if PolPercentage else pol
        
        if self.class_QS.PropagationSpace == "Liouville":
            rho = self.class_QS.Class_quantumlibrary.VecToDM(rhoQ, (self.class_QS.Vdim,self.class_QS.Vdim)).data
            Sz = self.class_QS.Class_quantumlibrary.VecToDM(SzQ, (self.class_QS.Vdim,self.class_QS.Vdim)).data
            pol = -(1.0 / spinQ) * np.trace(rho @ Sz).real / np.trace(rho).real
            return 100 * pol if PolPercentage else pol            

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Matrix Functions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Create_DensityMatrix(self, state):
        """Create a density matrix from a pure state."""
        return np.outer(state, self.Adjoint(state))

    def Norm_Matrix(self, A):
        """Compute the Frobenius norm of a matrix."""
        return np.linalg.norm(A, ord='fro')

    def Adjoint(self, A):
        """Return the Hermitian adjoint (conjugate transpose) of a matrix."""
        return A.T.conj()

    def InnerProduct(self, A, B):
        """Compute the inner product ⟨A|B⟩."""
        return np.trace(self.Adjoint(A) @ B)

    def Normalize(self, A):
        """Return a normalized operator with unit inner product."""
        return A / np.sqrt(self.InnerProduct(A, A))

    def DensityMatrix_Components(self, AQ, dic, rhoQ, tol=1.0e-10, roundto=5):
        """
        Decompose a density matrix into a linear combination of a given operator basis.

        This function calculates the components of the density matrix with respect to a 
        specified basis of operators using an inner product. It prints the resulting 
        decomposition in a readable format.

        Parameters
        ----------
        AQ : list of QunObj
            List of basis operator objects that define the decomposition space.
        dic : dict
            Dictionary mapping indices to basis labels for readable output.
        rhoQ : QunObj
            The density matrix (or state vector) to be decomposed.
        tol : float, optional
            Tolerance level for treating small component values as zero. Default is 1.0e-10.
        roundto : int, optional
            Number of decimal places to round the non-zero components. Default is 5.

        Raises
        ------
        TypeError
            If `AQ` is not a list or contains elements that are not instances of `QunObj`.

        Returns
        -------
        None
            The function prints the decomposition of the density matrix but does not return it.
        """

        # Check if AQ is a list and contains only QunObj instances
        if not isinstance(AQ, list):
            raise TypeError("Input must be a list.")
        if not all(isinstance(item, QunObj) for item in AQ):
            raise TypeError("All elements must be instances of QunObj.")

        # Extract raw data from QunObj
        rho = rhoQ.data

        # Convert vector to density matrix if needed
        if rho.shape[1] == 1:
            rho = QLib.VecToDM(QunObj(rho), AQ[0].data.shape).data

        # Calculate inner products with basis elements
        components = np.array([self.InnerProduct(A.data, rho) for A in AQ])

        # Zero out small real and imaginary parts
        components.real[abs(components.real) < tol] = 0.0
        components.imag[abs(components.imag) < tol] = 0.0

        # Build the string representation of the decomposition
        output = ["Density Matrix = "]
        for i, val in enumerate(components):
            if val.real != 0:
                output.append(f"{round(val.real, roundto)} {dic[i]} + ")

        # Print result, removing the trailing ' + '
        print((''.join(output))[:-3])


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Liouville Vectors
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Vector_L(self, X):
        """
        Vectorize an operator into a Liouville space column vector.

        Parameters
        ----------
        X : ndarray
            Operator matrix.

        Returns
        -------
        ndarray
            Vectorized form.
        """
        dim = self.class_QS.Vdim
        return np.reshape(X, (dim**2, -1))

    def Detection_L(self, X):
        """
        Detection vector for Liouville space.

        Parameters
        ----------
        X : ndarray
            Operator matrix.

        Returns
        -------
        ndarray
            Row vector (bra) for detection.
        """
        return self.Vector_L(X).conj().T

    def ProductOperators_ConvertToLiouville(self, Basis_X):
        """
        Convert basis operators to Liouville space.

        Parameters
        ----------
        Basis_X : list of ndarrays
            Basis operators in Hilbert space.

        Returns
        -------
        list of QunObj
            Operators in Liouville space.
        """
        return [QunObj(self.Vector_L(np.asarray(A))) for A in Basis_X]

    def Liouville_Bracket(self, A, B, C):
        """
        Compute the Liouville bracket ⟨A|B|C⟩.

        Parameters
        ----------
        A, B, C : ndarrays

        Returns
        -------
        float
            Real part of the bracket.
        """
        return np.trace(self.Adjoint(A) @ B @ C).real
