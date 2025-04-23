"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contains the class QunObj

QunObj - Quantum Object:
------------------------
A flexible class representing quantum states (kets, bras) and operators. 
It supports standard quantum operations such as Hermitian conjugation, 
tensor products, expectation values, and more.

Documentation is done.
"""

# ---------- Package Imports ----------

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

# ---------- Class Definition ----------

class QunObj():
    def __init__(self, data, Type=None, DType=complex, PrintDefault=False, tolerence=1.0e-10):
        """
        Initialize a quantum object.

        Parameters
        ----------
        data : array-like
            The matrix or vector data (NumPy-compatible) representing a quantum object.
        Type : str, optional
            Type of quantum object ('ket', 'bra', 'operator'). If not specified, inferred from shape.
        DType : data-type, optional
            The data type for matrix elements. Default is complex.
        PrintDefault : bool, optional
            Whether to print object info on creation.
        tolerence : float, optional
            Threshold below which real or imaginary parts are zeroed out.
        
        Attributes
        ----------
        data : np.ndarray
            The cleaned-up internal matrix representation.
        shape : tuple
            The shape of the object.
        datatype : np.dtype
            Data type of the stored array.
        type : str
            Object type: 'ket', 'bra', or 'operator'.
        matrix : sympy.Matrix
            Symbolic version of the matrix (for display or symbolic analysis).
        Matrix_tol : float
            Numerical tolerance threshold.
        """

        self.Matrix_tol = tolerence  # Threshold for zeroing small values

        # Convert to NumPy array and clean near-zero components
        Matrix_copy = (np.array(data, dtype=DType)).copy()
        Matrix_copy.real[abs(Matrix_copy.real) < self.Matrix_tol] = 0.0
        Matrix_copy.imag[abs(Matrix_copy.imag) < self.Matrix_tol] = 0.0

        self.data = Matrix_copy                      # Numerical data
        self.shape = self.data.shape                 # Matrix shape
        self.datatype = self.data.dtype              # Data type of elements
        self.matrix = sp.Matrix(self.data.tolist())  # SymPy symbolic representation

        # Determine object type (ket, bra, operator)
        if Type is None:
            if self.shape[1] == 1:
                self.type = "ket"
            elif self.shape[0] == 1:
                self.type = "bra"
            else:
                self.type = "operator"
        else:
            self.type = Type

        if PrintDefault:
            print(f"Quantum object initialized: shape={self.shape}, type='{self.type}', dtype={self.datatype}")

    def Adjoint(self):
        """
        Return the Hermitian conjugate (dagger) of the quantum object.

        Returns
        -------
        QunObj
            The conjugate transpose of the object.
        """
        new_type = "bra" if self.type == "ket" else "ket" if self.type == "bra" else "operator"
        return QunObj(self.data.conj().T, Type=new_type)

    def Conjugate(self):
        """
        Return the complex conjugate of the quantum object.

        Returns
        -------
        QunObj
            The complex conjugated quantum object.
        """
        return QunObj(self.data.conj(), Type=self.type)

    def Tranpose(self):
        """
        Return the transpose of the quantum object (without complex conjugation).

        Returns
        -------
        QunObj
            The transposed quantum object.
        """
        new_type = "bra" if self.type == "ket" else "ket" if self.type == "bra" else "operator"
        return QunObj(self.data.T, Type=new_type)

    def Inverse(self):
        """
        Compute the matrix inverse.

        Returns
        -------
        QunObj
            The inverse of the quantum object.

        Raises
        ------
        np.linalg.LinAlgError
            If the matrix is singular or not square.
        """
        return QunObj(np.linalg.inv(self.data), Type=self.type)

    def Inverse2PI(self):
        """
        Divide all elements of the quantum object by 2π.

        Useful for converting angular frequency to frequency units.

        Returns
        -------
        QunObj
            The scaled object.
        """
        return QunObj(self.data / (2.0 * np.pi), Type=self.type)

    def Trace(self):
        """
        Compute the trace of the matrix.

        Returns
        -------
        complex
            The trace value.
        """
        return np.trace(self.data)

    def NeumannEntropy(self):
        """
        Compute the Von Neumann entropy S = -Tr(ρ log ρ).

        Returns
        -------
        float
            The entropy value.
        """
        return -1 * np.trace(self.data @ np.log(self.data))

    def Norm(self):
        """
        Compute the Frobenius norm of the matrix.

        Returns
        -------
        float
            Frobenius norm ||A||_F.
        """
        return np.linalg.norm(self.data, ord='fro')

    def Norm_HZ(self):
        """
        Compute the Frobenius norm scaled by 1 / (2π).

        Returns
        -------
        float
            Scaled Frobenius norm.
        """
        return self.Norm() / (2 * np.pi)

    def Purity(self):
        """
        Compute the purity Tr(ρ²) of a density matrix.

        Returns
        -------
        float
            Purity value (1 for pure states).
        """
        return np.trace(np.matmul(self.data, self.data))

    def Expm(self):
        """
        Compute the matrix exponential exp(A).

        Returns
        -------
        QunObj
            The matrix exponential of the object.
        """
        return QunObj(la.expm(self.data), Type=self.type)

    def Hermitian(self):
        """
        Check whether the matrix is Hermitian (A = A†).

        Returns
        -------
        bool
            True if Hermitian, False otherwise.
        """
        return np.allclose(self.data, self.data.conj().T)

    def Positive(self):
        """
        Print whether all diagonal elements are non-negative.

        Useful for checking if a density matrix is physical.
        """
        has_negative_diagonal = np.any(np.diag(self.data) < 0)
        print("False" if has_negative_diagonal else "True")

    def __add__(self, *others):
        """
        Add multiple QunObj instances element-wise.

        Parameters
        ----------
        *others : QunObj
            Quantum objects to add.

        Returns
        -------
        QunObj
            Resulting sum.

        Raises
        ------
        ValueError
            If shapes do not match.
        """
        result = self
        for other in others:
            if isinstance(other, QunObj) and self.shape == other.shape:
                result = QunObj(result.data + other.data)
            else:
                raise ValueError("Addition requires matching quantum object shapes.")
        return result

    def __sub__(self, *others):
        """
        Subtract multiple QunObj instances element-wise.

        Parameters
        ----------
        *others : QunObj
            Quantum objects to subtract.

        Returns
        -------
        QunObj
            Resulting difference.

        Raises
        ------
        ValueError
            If shapes do not match.
        """
        result = self
        for other in others:
            if isinstance(other, QunObj) and self.shape == other.shape:
                result = QunObj(result.data - other.data)
            else:
                raise ValueError("Subtraction requires matching quantum object shapes.")
        return result

    def __mul__(self, other):
        """
        Multiply this QunObj with a scalar or another QunObj.

        Parameters
        ----------
        other : QunObj or scalar
            Right-hand operand.

        Returns
        -------
        QunObj
            Resulting product.

        Raises
        ------
        ValueError or TypeError
            If dimensions are incompatible or type unsupported.
        """
        if isinstance(other, (int, float, complex)):
            return QunObj(self.data * other)
        elif isinstance(other, QunObj):
            if self.shape[1] == other.shape[0]:
                return QunObj(np.matmul(self.data, other.data))
            else:
                raise ValueError("Matrix multiplication: incompatible dimensions.")
        else:
            raise TypeError("Multiplication only supports scalars or QunObj.")

    def __rmul__(self, other):
        """
        Scalar multiplication with scalar on the left.

        Parameters
        ----------
        other : scalar
            Scalar value.

        Returns
        -------
        QunObj
        """
        if isinstance(other, (int, float, complex)):
            return QunObj(self.data * other)
        else:
            return NotImplemented

    def __truediv__(self, scalar):
        """
        Divide all elements by a scalar.

        Parameters
        ----------
        scalar : float or complex

        Returns
        -------
        QunObj

        Raises
        ------
        ZeroDivisionError or TypeError
        """
        if isinstance(scalar, (int, float, complex)):
            if scalar == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return QunObj(self.data / scalar)
        else:
            raise TypeError("Division only supports scalars.")

    def Commute(self, other):
        """
        Check if two QunObj instances commute.

        Parameters
        ----------
        other : QunObj

        Returns
        -------
        bool
            True if commutator is zero, False otherwise.
        """
        if not isinstance(other, QunObj):
            raise TypeError("Commute only supports other QunObj instances.")
        commutator = self.data @ other.data - other.data @ self.data
        result = np.allclose(commutator, 0)
        print("Commute" if result else "Don't Commute")
        return result

    def TensorProduct(self, other):
        """
        Compute tensor product with another QunObj.

        Parameters
        ----------
        other : QunObj

        Returns
        -------
        QunObj
        """
        if not isinstance(other, QunObj):
            raise TypeError("Tensor product only supports QunObj.")
        return QunObj(np.kron(self.data, other.data))

    def OuterProduct(self, other):
        """
        Outer product: `|ψ⟩⟨φ|`.

        Parameters
        ----------
        other : QunObj

        Returns
        -------
        QunObj
        """
        if not isinstance(other, QunObj):
            raise TypeError("Outer product only supports QunObj.")
        return QunObj(np.outer(self.data, other.data.conj()))

    def InnerProduct(self, other):
        """
        Inner product: ⟨ψ|φ⟩ or Tr(A†B).

        Parameters
        ----------
        other : QunObj

        Returns
        -------
        complex
        """
        if not isinstance(other, QunObj):
            raise TypeError("Inner product only supports QunObj.")
        return np.trace(np.matmul(self.data.conj().T, other.data))

    def Normalize(self):
        """
        Normalize the quantum state or operator.

        Returns
        -------
        QunObj
            Normalized object.
        """
        norm = np.trace(self.data.conj().T @ self.data)
        return self if norm == 0 else QunObj(self.data / np.sqrt(norm))

    def Rotate(self, theta_rad, operator):
        r"""
        Apply a unitary rotation to the quantum object.

        For operators:
        :math:`\rho \rightarrow U \rho U^\dagger` where :math:`U = \exp(-i \theta A)`

        For states:
        :math:`|\psi\rangle \rightarrow U |\psi\rangle` where :math:`U = \exp(-i \theta A)`

        Parameters
        ----------
        theta_rad : float
            Rotation angle in degrees.
        operator : QunObj
            Hermitian operator generating the rotation.

        Returns
        -------
        QunObj
            Rotated quantum object.

        Raises
        ------
        TypeError
            If `operator` is not a QunObj.
        """

        if not isinstance(operator, QunObj):
            raise TypeError("Rotate only supports QunObj as the operator.")

        theta_rad = np.pi * theta_rad / 180.0  # Convert degrees to radians
        U = expm(-1j * theta_rad * operator.data)

        if self.type == "operator":
            return QunObj(U @ self.data @ U.T.conj())
        elif self.type == "ket":
            return QunObj(U @ self.data)
        else:
            raise ValueError("Rotation only supported for 'ket' and 'operator' types.")

    def Expectation(self, operator):
        r"""
        Compute the expectation value of an operator.

        For a state :math:`|\psi\rangle`: :math:`\langle \psi | A | \psi \rangle`  
        For a density matrix :math:`\rho`: :math:`\mathrm{Tr}(\rho A)`

        Parameters
        ----------
        operator : QunObj
            Operator for which to compute the expectation value.

        Returns
        -------
        complex
            Expectation value.

        Raises
        ------
        TypeError
            If input is not a QunObj.
        """

        if not isinstance(operator, QunObj):
            raise TypeError("Expectation only supports QunObj as the operator.")

        if self.type == "operator":
            return np.trace(self.data @ operator.data)
        elif self.type == "ket":
            return np.trace(operator.data @ self.data)
        else:
            raise ValueError("Unsupported QunObj type for expectation value.")

    def Round(self, roundto):
        """
        Round the entries in the quantum object to a given decimal place.

        Parameters
        ----------
        roundto : int
            Number of decimal places to round.

        Returns
        -------
        QunObj
            Rounded object.
        """
        return QunObj(np.round(self.data, roundto))

    def Tolarence(self, tol):
        """
        Zero out all elements with magnitude below the specified tolerance.

        Parameters
        ----------
        tol : float
            Tolerance value.

        Returns
        -------
        QunObj
            Cleaned object with small elements set to zero.
        """
        Matrix_copy = self.data.copy()
        Matrix_copy.real[abs(Matrix_copy.real) < tol] = 0.0
        Matrix_copy.imag[abs(Matrix_copy.imag) < tol] = 0.0
        return QunObj(Matrix_copy)

