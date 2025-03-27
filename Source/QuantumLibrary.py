"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.coml

This file contain class QuantumLibrary

Attribute:
    ...

Methods:
    ...    

"""

# ---------- Package

from QuantumObject import QunObj
import PhysicalConstants 
import SpinQuantumNumber 
import Gamma
import QuadrupoleMoment 
import Particle

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


class QuantumLibrary():
    def __init__(self):
        self.hbarEQ1=True
        self.RowColOrder = "C"

        # Acquisition Parameters
        self.AcqDT = 0.0001
        self.AcqFS = 1.0/self.AcqDT
        self.AcqAQ = 5.0

        # ODE Parameters
        self.OdeMethod = 'RK45'
        self.ODE_atol = 1.0e-13
        self.ODE_rtol = 1.0e-13

    def Basis_Ket(self,dim,index,PrintDefault=False):
        """
        Create a column vector for a Hilbert space of dimension 'dim' with the state being the 'index' position set to 1, and the rest set to 0.
        """      

        if index < 0 or index >= dim:
            raise ValueError(f"Index must be between 0 and {dim - 1}.")

        # Create a column vector with 'dim' rows and 'index'-th element set to 1
        state = np.zeros((dim, 1), dtype=complex)
        state[index] = 1.0  # Set the specified index to 1 (all others are 0)
        
        return QunObj(state,PrintDefault=PrintDefault) # implement this to all PrintDefault=PrintDefault

    def Basis_Bra(self,dim,index,PrintDefault=False):
        """
        Create a column vector for a Hilbert space of dimension 'dim' with the state being the 'index' position set to 1, and the rest set to 0.
        """      

        if index < 0 or index >= dim:
            raise ValueError(f"Index must be between 0 and {dim - 1}.")

        # Create a column vector with 'dim' rows and 'index'-th element set to 1
        state = np.zeros((1,dim), dtype=complex)
        state[0, index] = 1.0  # Set the specified index to 1 (all others are 0)
        
        return QunObj(state,PrintDefault=PrintDefault)
    
    def Bloch_Vector(self, theta, phi):
        """
        Generate the Bloch Vector
        theta and phi the shherical coordinate
        """

        vec1 = self.Basis_Ket(2,0,PrintDefault=False)
        vec2 = self.Basis_Ket(2,1,PrintDefault=False)

        theta = np.pi/180.0 * theta
        phi = np.pi/180.0 * phi

        return np.cos(theta/2) * vec1 + np.exp(1j * phi) * np.sin(theta/2) * vec2

    def SSpinOp(self,X,String,PrintDefault=False):
        """
        Single Spin operators
        """   
        
        if self.hbarEQ1:
            hbar = 1
        else:
            hbar = 1.054e-34    
            
        # 1D Array: magnetic qunatum number for spin S (order: S, S-1, ... , -S)
        ms = np.arange(X,-X-1,-1)  
        
        # Initialize Sx, Sy and Sz operators for a spin, S
        SingleSpin = np.zeros((3,ms.shape[-1],ms.shape[-1]),dtype=np.csingle,order="C")
        
        # Intitialze S+ and S- operators for spin, S
        Sp = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=np.csingle,order="C")
        Sn = np.zeros((ms.shape[-1],ms.shape[-1]),dtype=np.csingle,order="C")
        
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
                SingleSpin[2][i][j] = hbar * ms[j]*Id[i][j] 
                
                # S+ operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S>  
                Sp[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]+1)) * Idp[i][j] 
                # S- operator, Row ordering (top to bottom): |j,S>, |j,S-1>,... , |j,-S> 
                Sn[i][j] = np.sqrt(X*(X+1) - ms[j]*(ms[j]-1)) * Idn[i][j] 
        
        # Sx operator
        SingleSpin[0] = hbar * (1/2.0) * (Sp + Sn) 
        # Sy operator
        SingleSpin[1] = hbar * (-1j/2.0) * (Sp - Sn) 

        if String == "x":
            return QunObj(SingleSpin[0],PrintDefault=PrintDefault)

        if String == "y":
            return QunObj(SingleSpin[1],PrintDefault=PrintDefault)

        if String == "z":
            return QunObj(SingleSpin[2],PrintDefault=PrintDefault)
        
        if String == "p":
            return QunObj(Sp,PrintDefault=PrintDefault)    
        
        if String == "m":
            return QunObj(Sn,PrintDefault=PrintDefault)      


    def Purity(self, A: QunObj) -> QunObj:
        """
        Compute the tensor product of two QunObj instances.
        """
        if not isinstance(A, QunObj):
            raise TypeError("Purity only supports QunObj instances.")

        # Compute the tensor product using numpy's kron function
        data = np.matmul(A.data, A.data)
        
        # Return the resulting tensor product as a new QunObj instance
        return QunObj(np.trace(data))
    
    def TensorProduct(self, A: QunObj, B: QunObj) -> QunObj:
        """
        Compute the tensor product of two QunObj instances.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Tensor product only supports QunObj instances.")

        # Compute the tensor product using numpy's kron function
        tensor_data = np.kron(A.data, B.data)
        
        # Return the resulting tensor product as a new QunObj instance
        return QunObj(tensor_data)

    def TensorProductMultiple(self, *matrices: QunObj) -> QunObj:
        """
        Compute the tensor product of multiple QunObj instances.
        result = reduce(np.kron, [A, B, C]); np.kron(A, B); np.kron(result, C)
        
        Parameters:
            *matrices: Variable number of QunObj instances.
        
        Returns:
            A new QunObj instance representing the tensor product.
        """
        if not matrices:
            raise ValueError("At least one QunObj must be provided.")

        if not all(isinstance(mat, QunObj) for mat in matrices):
            raise TypeError("Tensor product only supports QunObj instances.")

        # Compute the tensor product iteratively using functools.reduce
        result_data = reduce(np.kron, (mat.data for mat in matrices))
        
        # Return as a new QunObj
        return QunObj(result_data)

    def DirectSum(self, A: QunObj, B: QunObj) -> QunObj:
        """
        Compute the direct sum of two QunObj instances.

        Parameters:
            A: First QunObj.
            B: Second QunObj.

        Returns:
            A new QunObj instance representing the direct sum.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")
        
        # If both are kets, return a concatenation
        if A.type == "ket" and B.type == "ket":
            return QunObj(np.vstack((A.data, B.data)), Type="ket")
        
        # If both are matrices, perform the direct sum
        if A.type == "operator" and B.type == "operator":
            # Get dimensions
            rows_A, cols_A = A.data.shape
            rows_B, cols_B = B.data.shape

            # Create a zero-padded matrix for the direct sum
            result = np.zeros((rows_A + rows_B, cols_A + cols_B), dtype=A.data.dtype)

            # Fill in diagonal blocks
            result[:rows_A, :cols_A] = A.data
            result[rows_A:, cols_A:] = B.data

            return QunObj(result, Type="operator")
        
        raise ValueError("Direct sum is only defined for two kets or two matrices.")

    def DirectSumMultiple(self, *matrices: QunObj) -> QunObj:
        """
        Compute the direct sum of multiple QunObj instances.

        Parameters:
            *matrices: Variable number of QunObj instances.

        Returns:
            A new QunObj instance representing the direct sum.
        """
        if not matrices:
            raise ValueError("At least one QunObj must be provided.")

        if not all(isinstance(mat, QunObj) for mat in matrices):
            raise TypeError("Direct sum only supports QunObj instances.")
        
        # Check if all instances are kets or all are operators
        types = {mat.type for mat in matrices}
        if len(types) > 1:
            raise ValueError("Direct sum is only defined for kets or operators, not a mix of both.")
        
        if "ket" in types:
            return QunObj(np.vstack([mat.data for mat in matrices]), Type="ket")
        
        if "operator" in types:
            total_rows = sum(mat.data.shape[0] for mat in matrices)
            total_cols = sum(mat.data.shape[1] for mat in matrices)
            
            result = np.zeros((total_rows, total_cols), dtype=matrices[0].data.dtype)
            
            row_offset, col_offset = 0, 0
            for mat in matrices:
                rows, cols = mat.data.shape
                result[row_offset:row_offset + rows, col_offset:col_offset + cols] = mat.data
                row_offset += rows
                col_offset += cols
            
            return QunObj(result, Type="operator")
        
        raise ValueError("Unsupported QunObj type for direct sum.")

    def OuterProduct(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the outer product of two QunObj instances.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Outer product only supports QunObj instances.")

        # Compute the outer product using np.outer
        outer_product_data = np.outer(A.data, B.data.conj())  # Ensures correct complex conjugation

        # Return the resulting outer product as a new QunObj instance
        return QunObj(outer_product_data)

    def InnerProduct(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the outer product of two QunObj instances.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Inner product only supports QunObj instances.")

        # Compute the inner product
        inner_product_data = np.matmul(A.data.conj().T, B.data)  

        # Return the resulting inner product as a new QunObj instance
        return inner_product_data
    
    def Identity(self,dim):
        """
        Identity Matrix
        """            

        return QunObj(np.eye(dim))
    
    def Commutator(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Commutator
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Commutator only supports QunObj instances.")

        # Compute the inner product
        commut = np.matmul(A.data, B.data) - np.matmul(B.data, A.data) 

        # Return the resulting commutator as a new QunObj instance
        return QunObj(commut)  

    def Commutator_Array(self, A, B):
        """
        Commutator
        """

        # Compute the inner product
        commut = np.matmul(A, B) - np.matmul(B, A) 

        # Return the resulting commutator as a new QunObj instance
        return commut

    def DoubleCommutator(self, A: 'QunObj', B: 'QunObj', rho: 'QunObj') -> 'QunObj':
        """
        Commutator
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj) or not isinstance(rho, QunObj):
            raise TypeError("DoubleCommutator only supports QunObj instances.")

        # Compute the inner product
        dcommut = np.matmul(B.data, rho.data) - np.matmul(rho.data, B.data) 
        commut = np.matmul(A.data, dcommut) - np.matmul(dcommut, A.data)

        # Return the resulting commutator as a new QunObj instance
        return QunObj(commut)  

    def AntiCommutator(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Commutator
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Anti Commutator only supports QunObj instances.")

        # Compute the inner product
        anticommut = np.matmul(A.data, B.data) + np.matmul(B.data, A.data) 

        # Return the resulting anti commutator as a new QunObj instance
        return QunObj(anticommut) 

    def PartialTrace(self, rho: 'QunObj', Sdim, keep)-> 'QunObj':
        """
        Partial Trace (Testing)
        
        INPUT
        -----
        rho  : densithy matrix
        Sdim  : list of individual
        keep : list subsystem to keep
        Sdim : Dimension of individual
        
        OUTPUT
        ------
        """

        if not isinstance(rho, QunObj):
            raise TypeError("rho only supports QunObj instances.")

        SysInx = range(len(Sdim)) # Indices of all subsystem
        TraceInx = list(set(SysInx) - set(keep)) # Subsystem to traced out
        
        new_shape = Sdim + Sdim
        rho_new = rho.data.reshape(new_shape)
        
        for idx in sorted(TraceInx, reverse=True):
            rho_new = np.trace(rho_new, axis1= idx, axis2= idx + len(rho_new.shape) // 2)
            
        Sdim_new = [Sdim[j] for j in keep]  
        final_shape = (np.prod(Sdim_new), np.prod(Sdim_new)) 
        
        return QunObj(rho_new.reshape(final_shape))  

    def BlockExtract(self, Matrix: 'QunObj', block_index, block_sizes) -> 'QunObj':
        """
        Extracts a specific block from a block diagonal matrix with blocks of varying sizes,
        and for ket matrices, reshapes the ket as column vectors based on the block sizes.
        
        Parameters:
        - Matrix (QunObj): The block diagonal matrix.
        - block_index (int): The index of the block to extract (0-based index).
        - block_sizes (list of tuples): A list of tuples where each tuple contains
                                        the dimensions of the corresponding block (rows, cols).
        
        Returns:
        - QunObj: The extracted block matrix or reshaped ket vector.
        
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html
        """

        if not isinstance(Matrix, QunObj):
            raise TypeError("Matrix only supports QunObj instances.")
        
        matrix = Matrix.data
        
        # Handle case where the matrix type is "operator"
        if Matrix.type == "operator":
            # Ensure the matrix dimensions match the sum of the blocks' dimensions
            total_rows = sum(rows for rows, _ in block_sizes)
            total_cols = sum(cols for _, cols in block_sizes)

            if matrix.shape[0] != total_rows or matrix.shape[1] != total_cols:
                raise ValueError("Matrix size does not match the sum of block sizes.")
            
            # Calculate the row and column positions for the block
            start_row = sum(block_sizes[i][0] for i in range(block_index))
            end_row = start_row + block_sizes[block_index][0]
            
            start_col = sum(block_sizes[i][1] for i in range(block_index))
            end_col = start_col + block_sizes[block_index][1]
            
            # Extract the block by slicing the matrix
            block_matrix = matrix[start_row:end_row, start_col:end_col]
            
            return QunObj(block_matrix, Type="operator")
        
        # Handle case where the matrix type is "ket"
        elif Matrix.type == "ket":
            # Flatten the ket vector into a 1D array
            flat_matrix = matrix.flatten()  # Flatten the ket into a 1D array
            
            # Calculate the start and end indices for the desired block
            start_index = sum(block_sizes[i][0] for i in range(block_index))  # Sum of all previous rows' sizes
            end_index = start_index + block_sizes[block_index][0]  # The end index for the block

            # Extract the corresponding block from the flattened ket vector
            block_vector = flat_matrix[start_index:end_index]
            
            # Reshape the block into a column vector
            reshaped_block = np.reshape(block_vector, (-1, 1))  # Reshape as a column vector
            
            return QunObj(reshaped_block, Type="ket")
        
        # If the matrix type is not recognized
        else:
            raise ValueError(f"Unsupported matrix type: {Matrix.type}")

    def DMToVec(self, rho: 'QunObj')-> 'QunObj': 
        """
        Density matrix to vector
        """
        if not isinstance(rho, QunObj):
            raise TypeError("rho only supports QunObj instances.")  

        if self.RowColOrder == 'C':
            return QunObj(rho.data.flatten('C').reshape(-1, 1))  # 'C' ensures row-major order   

        if self.RowColOrder == 'F':
            return QunObj(rho.data.flatten('F').reshape(-1, 1))  # 'F' ensures column-major order

    def VecToDM(self, vec: 'QunObj', shape: tuple) -> 'QunObj':
        """
        Vector to density matrix
        """
        if not isinstance(vec, QunObj):
            raise TypeError("vec only supports QunObj instances.")
        
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("shape must be a tuple of length 2.")
        
        if self.RowColOrder == 'C':
            return QunObj(vec.data.reshape(shape, order='C'))  # Reshape in row-major order
        
        if self.RowColOrder == 'F':
            return QunObj(vec.data.reshape(shape, order='F'))  # Reshape in column-major order  

    def CommutationSuperoperator(self, X: 'QunObj') -> 'QunObj':
        """
        Computes the commutation superoperator: [H, rho] = H rho - rho H
        
        Parameters:
        -----------
        X : QunObj
            Input quantum object representing the operator H.

        Returns:
        --------
        QunObj
            Commutation superoperator as a QunObj instance.
        """     
        if not isinstance(X, QunObj):
            raise TypeError("Input must be an instance of QunObj.")
        
        identity_matrix = np.eye(X.shape[-1])

        if self.RowColOrder == 'C':        
            commutator_matrix = np.kron(X.data, identity_matrix) - np.kron(identity_matrix, X.data.T)

        if self.RowColOrder == 'F':        
            commutator_matrix = np.kron(identity_matrix, X.data) - np.kron(X.data.T, identity_matrix)             
        
        return QunObj(commutator_matrix)   

    def AntiCommutationSuperoperator(self, X: 'QunObj') -> 'QunObj':
        """
        Computes the anti commutation superoperator: [H, rho] = H rho - rho H
        
        Parameters:
        -----------
        X : QunObj
            Input quantum object representing the operator H.

        Returns:
        --------
        QunObj
            Anti Commutation superoperator as a QunObj instance.
        """     
        if not isinstance(X, QunObj):
            raise TypeError("Input must be an instance of QunObj.")
        
        identity_matrix = np.eye(X.shape[-1])

        if self.RowColOrder == 'C':        
            commutator_matrix = np.kron(X.data, identity_matrix) + np.kron(identity_matrix, X.data.T)

        if self.RowColOrder == 'F':        
            commutator_matrix = np.kron(identity_matrix, X.data) + np.kron(X.data.T, identity_matrix)             
        
        return QunObj(commutator_matrix) 

    def DoubleCommutationSuperoperator(self, X: 'QunObj', Y: 'QunObj') -> 'QunObj':
        """
        Computes the double commutation superoperator: [[X, rho], Y] = (X rho - rho X) Y - Y (X rho - rho X)
        
        Parameters:
        -----------
        X : QunObj
            First input quantum object representing an operator.
        Y : QunObj
            Second input quantum object representing an operator.

        Returns:
        --------
        QunObj
            Double commutation superoperator as a QunObj instance.
        """     
        if not isinstance(X, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")
        
        identity_X = np.eye(X.shape[-1])
        identity_Y = np.eye(Y.shape[-1])
        
        if self.RowColOrder == 'C':
            commutator_X = np.kron(X.data, identity_X) - np.kron(identity_X, X.data.T)
            commutator_Y = np.kron(Y.data, identity_Y) - np.kron(identity_Y, Y.data.T)
        
        if self.RowColOrder == 'F':
            commutator_X = np.kron(identity_X, X.data) - np.kron(X.data.T, identity_X)
            commutator_Y = np.kron(identity_Y, Y.data) - np.kron(Y.data.T, identity_Y)
        
        double_commutator = np.matmul(commutator_X, commutator_Y)
        
        return QunObj(double_commutator)  

    def Bracket(self, X: 'QunObj', A: 'QunObj', Y: 'QunObj') -> 'QunObj':
        """
        Computes the double commutation superoperator: [[X, rho], Y] = (X rho - rho X) Y - Y (X rho - rho X)
        
        Parameters:
        -----------
        X : QunObj
            First input quantum object representing an operator.
        Y : QunObj
            Second input quantum object representing an operator.

        Returns:
        --------
        QunObj
            Double commutation superoperator as a QunObj instance.
        """     
        if not isinstance(X, QunObj) or not isinstance(A, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")
        
        return np.trace(np.matmul(X.data.conj().T,np.matmul(A.data,Y.data)))

    def Eigen(self, A: QunObj) -> QunObj:   
        """
        Eigen values and eigen vector
        """        

        if not isinstance(A, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")

        eigenvalues, eigenvectors = la.eig(A.data)  

        return QunObj(eigenvalues.reshape(1, -1) .real), QunObj(eigenvectors)

    def Eigen_Split(self, A: QunObj) -> QunObj:   
        """
        Eigenvalues and eigenvectors, where eigenvectors are split into individual QunObj instances.
        """        

        if not isinstance(A, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = la.eig(A.data)  

        # Convert eigenvalues to QunObj instance
        eigenvalue_obj = QunObj(eigenvalues.reshape(1, -1).real)

        # Split eigenvectors as individual QunObj instances, each a column vector
        eigenvector_objs = [QunObj(vec.reshape(-1, 1)) for vec in eigenvectors.T]

        return eigenvalue_obj, eigenvector_objs

    def Evolve_SE_UProp(self,vec,Hamiltonian):
        """
        
        """

        if not isinstance(Hamiltonian, QunObj):
            raise TypeError("Hamiltonian only supports QunObj instances.")

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        vec_ = vec.data
        vec_t = [QunObj(vec_)]
        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
        U = expm(-1j * Hamiltonian.data * dt)
        
        for i in range(Npoints):
            vec_ = np.matmul(U,vec_)
            vec_t.append(QunObj(vec_))
        return t, vec_t    

    def Evolve_SE_ODE(self,vec,Hamiltonian):
        """
        
        """

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        vec_ = vec.data
        vec_t = [QunObj(vec_)]

        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)

        Lvec = vec_.flatten().astype(complex)  # Ensure it's a 1D complex array
        t = np.linspace(0, dt * Npoints, Npoints, endpoint=True)

        def vecDOT(t, Lvec, Hamiltonian):
            return -1j * Hamiltonian.data @ Lvec  # No need for redundant reshaping
    
        vecSol = solve_ivp(vecDOT,[0,dt*Npoints],Lvec,method=self.OdeMethod,t_eval=t,args=(Hamiltonian,), atol = self.ODE_atol, rtol = self.ODE_rtol)   
        t, vec_sol = vecSol.t, vecSol.y

        for i in range(Npoints):
            vec_t.append(QunObj(np.reshape(vec_sol[:,i],(vec_.shape[0],1))))
                                
        return t, vec_t        

    def Evolve_Hilbert_UProp(self,rho,Hamiltonian):
        """
        
        """

        if not isinstance(Hamiltonian, QunObj):
            raise TypeError("Hamiltonian only supports QunObj instances.")

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        rho_ = rho.data
        rho_t = [QunObj(rho_)]
        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
        U = expm(-1j * Hamiltonian.data * dt)
        for i in range(Npoints):
            rho_ = np.matmul(U,np.matmul(rho_,U.T.conj()))
            rho_t.append(QunObj(rho_))
        return t, rho_t

    def Evolve_Hilbert_ODE(self,rho,Hamiltonian):
        """
        
        """

        dt = self.AcqDT
        Npoints = int(self.AcqAQ / self.AcqDT)

        rho_ = rho.data  # Extract matrix from QunObj
        H = Hamiltonian.data  # Extract Hamiltonian matrix from QunObj

        # Ensure Hamiltonian is a square 2D array
        H = np.array(H, dtype=complex)
        if H.shape[0] != H.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix.")

        # Flatten density matrix for solve_ivp
        rhoi = rho_.reshape(-1).astype(complex)

        # Time points for integration
        t = np.linspace(0, dt * Npoints, Npoints, endpoint=True)

        def rhoDOT(t, rhoi, H):
            """Computes the derivative of the density matrix."""
            rho_temp = rhoi.reshape(rho_.shape)
            rhodot = -1j * self.Commutator_Array(H, rho_temp)  # Compute commutator
            return rhodot.reshape(-1)  # Flatten for ODE solver

        # Solve the ODE
        rhoSol = solve_ivp(
            rhoDOT, [0, dt * Npoints], rhoi, method=self.OdeMethod, t_eval=t, 
            args=(H,), atol=self.ODE_atol, rtol=self.ODE_rtol
        )

        t, rho2d = rhoSol.t, rhoSol.y

        # Reconstruct density matrices at each time step
        rho_t = [QunObj(rho2d[:, i].reshape(rho_.shape)) for i in range(Npoints)]

        return t, rho_t               

    def Evolve_Liouville_UProp(self,vec,Hamiltonian):
        """
        
        """

        if not isinstance(Hamiltonian, QunObj):
            raise TypeError("Hamiltonian only supports QunObj instances.")

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        vec_ = vec.data
        vec_t = [QunObj(vec_)]
        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)
        U = expm(-1j * Hamiltonian.data * dt)
        
        for i in range(Npoints):
            vec_ = np.matmul(U,vec_)
            vec_t.append(QunObj(vec_))
        return t, vec_t

    def Evolve_Liouville_ODE(self,vec,Hamiltonian):
        """
        
        """

        dt = self.AcqDT
        Npoints = int(self.AcqAQ/self.AcqDT)

        vec_ = vec.data
        vec_t = [QunObj(vec_)]

        t = np.linspace(0,dt*Npoints,Npoints,endpoint=True)

        Lvec = vec_.flatten().astype(complex)  # Ensure it's a 1D complex array
        t = np.linspace(0, dt * Npoints, Npoints, endpoint=True)

        def vecDOT(t, Lvec, Hamiltonian):
            return -1j * Hamiltonian.data @ Lvec  # No need for redundant reshaping
    
        vecSol = solve_ivp(vecDOT,[0,dt*Npoints],Lvec,method=self.OdeMethod,t_eval=t,args=(Hamiltonian,), atol = self.ODE_atol, rtol = self.ODE_rtol)   
        t, vec_sol = vecSol.t, vecSol.y

        for i in range(Npoints):
            vec_t.append(QunObj(np.reshape(vec_sol[:,i],(vec_.shape[0],1))))
                                
        return t, vec_t 
    
    def Expectation(self,t,matrix, operator):
        """
        
        """
        Npoints = len(t)
        signal = np.zeros(Npoints,dtype=complex)

        if matrix[0].type == "ket" and operator.type == "operator": # SE
            for i in range(Npoints):
                signal[i] = self.Bracket(matrix[i],operator,matrix[i])
        if matrix[0].type == "ket" and operator.type == "bra": # Liouville
            for i in range(Npoints):
                signal[i] = np.trace(operator.data @ matrix[i].data)                
        if matrix[0].type == "operator" and operator.type == "operator": # Hilbert
            for i in range(Npoints):
                signal[i] = np.trace(np.matmul(matrix[i].data,operator.data))

        return t, signal