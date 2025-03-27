"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.coml

This file contain class QunObj

Attribute:
    Matrix_tol
    data
    shape
    datatype
    type
    matrix

Methods:
    Adjoint
    Conjugate
    Purity
    Expm
    Tranpose
    Trace
    Norm
    Inverse
    Hermitian
    Positive
    __add__
    __mul__
    __rmul__
    Tensor
    OuterProduct
    InnerProduct
    Normalize
    Rotate
    Expectation
    Round
    Tolarence    

"""

# ---------- Package

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

class QunObj():
    def __init__(self, data, Type=None, DType=complex,PrintDefault=False):
        """
        Initialize a quantum object
        """

        self.Matrix_tol = 1.0e-10 # Make matrix element less 
        Matrix_copy = (np.array(data, dtype=DType)).copy()
        Matrix_copy.real[abs(Matrix_copy.real) < self.Matrix_tol] = 0.0
        Matrix_copy.imag[abs(Matrix_copy.imag) < self.Matrix_tol] = 0.0

        self.data = Matrix_copy # Attribute : Store matrix data as a NumPy array
        self.shape = self.data.shape # Attribute : Get shape of the matrix
        self.datatype = self.data.dtype # Attribute : Get the data type

        # Object type
        if Type is None:
            if self.shape[1] == 1:  # Column vector
                self.type = "ket"
            elif self.shape[0] == 1:  # Row vector
                self.type = "bra"
            else:
                self.type = "operator"
        else:
            self.type = Type  

        self.matrix = sp.Matrix(self.data.tolist()) # Attribute : Get symbolic matrix form

        if PrintDefault:
            print(f"Quantum object: shape={self.shape}, type='{self.type}', data type='{self.datatype}'")

    def Adjoint(self):
        """
        Method : Conjugate transpose
        """          
        return QunObj(self.data.conj().T, Type="bra" if self.type == "ket" else "ket")

    def Conjugate(self):
        """
        Method : Conjugate transpose
        """          
        return QunObj(self.data.conj())
    
    def Inverse2PI(self):
        """
        Method : divide by 2 PI
        """          
        return QunObj(self.data/(2.0 * np.pi))
        
    def Purity(self):
        """
        Purity of density matrix, mixedness
        """

        return np.trace(np.matmul(self.data,self.data))

    def Expm(self):
        """
        Purity of density matrix, mixedness
        """

        return QunObj(la.expm(self.data))

    def Tranpose(self):
        """
        Method : Conjugate transpose
        """          
        return QunObj(self.data.T, Type="bra" if self.type == "ket" else "ket")    

    def Trace(self):
        """
        Method : Trace
        """
        return np.trace(self.data)

    def Norm(self):
        """
        Frobenius norm
        """    
        return np.linalg.norm(self.data,ord='fro')
    
    def Inverse(self):
        """
        Inverse of a matrix
        """
        return QunObj(np.linalg.inv(self.data))
    
    def Hermitian(self):
        return np.allclose(self.data, self.data.conj().T)   

    def Positive(self):
        has_negative_diagonal = np.any(np.diag(self.data) < 0)
        if has_negative_diagonal:
            print("False")
        else:
            print("True")    
    
    def __add__(self, *others):
        """
        Addition of multiple quantum objects.
        """
        # Start with the current object (self)
        result = self
        
        for other in others:
            if isinstance(other, QunObj) and self.shape == other.shape:
                result = QunObj(result.data + other.data)  # Add the matrices element-wise
            else:
                raise ValueError("Addition requires matching quantum object shapes.")
        
        return result

    def __mul__(self, other):
        """
        Handle matrix multiplication and scalar multiplication when the scalar is on the right.
        """
        if isinstance(other, (int, float, complex)):  # Scalar multiplication (scalar after object)
            return QunObj(self.data * other)
        elif isinstance(other, QunObj):  # Matrix multiplication (QunObj × QunObj)
            if self.shape[1] == other.shape[0]:  # Valid matrix multiplication
                return QunObj(np.matmul(self.data, other.data))
            else:
                raise ValueError("Matrix multiplication: incompatible dimensions.")
        else:
            raise TypeError("Multiplication only supports scalars or QunObj.")

    def __rmul__(self, other):
        """
        Handle scalar multiplication when the scalar is on the left.
        """
        if isinstance(other, (int, float, complex)):  # Scalar multiplication (scalar before object)
            return QunObj(self.data * other)
        else:
            return NotImplemented  # Return NotImplemented to allow further fallback handling.    

    def Tensor(self, other):
        """
        Compute the tensor product of two quantum objects.
        """
        if not isinstance(other, QunObj):
            raise TypeError("Tensor product only supports other QunObj instances.")

        # Compute the tensor product using numpy's kron function
        tensor_data = np.kron(self.data, other.data)
        
        # The resulting object will have the shape of the combined dimensions
        return QunObj(tensor_data)   

    def OuterProduct(self, other):
        """
        Compute the outer product using np.outer (|ψ⟩⟨φ|).
        """
        if not isinstance(other, QunObj):
            raise TypeError("Outer product only supports other QunObj instances.")

        # Compute outer product using np.outer
        outer_product_data = np.outer(self.data, other.data.conj())

        return QunObj(outer_product_data)
    
    def InnerProduct(self, other):
        """
        Inner Product, vector and operator
        """    
        if not isinstance(other, QunObj):
            raise TypeError("Outer product only supports other QunObj instances.")

        # Compute outer product using np.outer
        inner_product_data = np.trace(np.matmul(self.data.conj().T, other.data))

        return inner_product_data

    def Normalize(self):
        """
        Normalize a column or row vector.
        """
        #norm = np.sqrt(np.vdot(self.data, self.data))  # Inner product ⟨v|v⟩
        norm = np.trace(np.matmul(self.data.conj().T, self.data))  # Inner product operator and vector
        if norm == 0:  # Avoid division by zero
            return self.data
        return QunObj(self.data / np.sqrt(norm))  # Normalize the vector 

    def Rotate(self,theta_rad,operator):
        """
        Rotation in Hilbert Space Operator    
        """  
        if not isinstance(operator, QunObj):
            raise TypeError("Outer product only supports other QunObj instances.")
        
        if self.type == "operator":                   
            theta_rad = np.pi * theta_rad / 180.0
            U = expm(-1j * theta_rad * operator.data)
            return QunObj(np.matmul(U,np.matmul(self.data,U.T.conj())))    

        if self.type == "ket":  
            theta_rad = np.pi * theta_rad / 180.0
            U = expm(-1j * theta_rad * operator.data)
            return QunObj(np.matmul(U,self.data))               

    def Expectation(self,operator):
        """
        Expectation value of an Operator    
        """  
        if not isinstance(operator, QunObj):
            raise TypeError("Outer product only supports other QunObj instances.")
        
        if self.type == "operator":                   
            return np.trace(np.matmul(self.data,operator.data))    

        if self.type == "ket":  
            return np.trace(operator.data @ self.data)
    
    def Round(self,roundto):
        """
        Round Matrix
        """
        
        return QunObj(np.round(self.data,roundto))
    
    def Tolarence(self,tol):
        """
        Round Matrix
        """
        
        Matrix_copy = self.data.copy()
        Matrix_copy.real[abs(Matrix_copy.real) < tol] = 0.0
        Matrix_copy.imag[abs(Matrix_copy.imag) < tol] = 0.0

        return QunObj(Matrix_copy)