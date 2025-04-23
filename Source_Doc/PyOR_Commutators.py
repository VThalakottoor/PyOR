"""
PyOR - Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
Email: vineethfrancis.physics@gmail.com

This file contains the class Commutators.

Documentation is done.
"""

import numpy as np
from scipy import sparse

try:
    from .PyOR_QuantumObject import QunObj  # For Sphinx/package context
except ImportError:
    from PyOR_QuantumObject import QunObj   # For direct script or notebook use



class Commutators:
    def __init__(self, class_QS=None):
        """
        Initialize the Commutators class.

        Parameters
        ----------
        class_QS : object, optional
            Quantum system object containing configuration.
        """
        if class_QS is not None:
            self.SparseM = class_QS.SparseM
            self.RowColOrder = class_QS.RowColOrder
        else:
            self.SparseM = False
            self.RowColOrder = 'C'

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Commutators and Superoperators
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Commutator(self, AQ, BQ):
        """
        Compute the commutator [A, B].

        Parameters
        ----------
        AQ, BQ : ndarray or object with .data
            Input matrices.

        Returns
        -------
        ndarray
            Commutator [A, B] = AB - BA
        """
        A = AQ.data if hasattr(AQ, 'data') else AQ
        B = BQ.data if hasattr(BQ, 'data') else BQ
        return np.matmul(A, B) - np.matmul(B, A)

    def DoubleCommutator(self, AQ, BQ, rhoQ):
        """
        Compute the double commutator [A, [B, ρ]].

        Parameters
        ----------
        AQ, BQ, rhoQ : ndarray or object with .data
            Input matrices.

        Returns
        -------
        ndarray
            Double commutator result.
        """
        A = AQ.data if hasattr(AQ, 'data') else AQ
        B = BQ.data if hasattr(BQ, 'data') else BQ
        rho = rhoQ.data if hasattr(rhoQ, 'data') else rhoQ
        return self.Commutator(A, self.Commutator(B, rho))

    def AntiCommutator(self, AQ, BQ):
        """
        Compute the anti-commutator {A, B}.

        Parameters
        ----------
        AQ, BQ : ndarray or object with .data
            Input matrices.

        Returns
        -------
        ndarray
            Anti-commutator {A, B} = AB + BA
        """
        A = AQ.data if hasattr(AQ, 'data') else AQ
        B = BQ.data if hasattr(BQ, 'data') else BQ
        return np.matmul(A, B) + np.matmul(B, A)

    def CommutationSuperoperator(self, XQ):
        """
        Construct the commutation superoperator [X, •].

        Parameters
        ----------
        XQ : ndarray or object with .data
            Operator matrix.

        Returns
        -------
        ndarray or sparse matrix
            Superoperator matrix.
        """
        X = np.array(XQ.data) if hasattr(XQ, 'data') else np.array(XQ)
        Id = np.identity(X.shape[-1])

        if self.RowColOrder == 'C':
            result = np.kron(X, Id) - np.kron(Id, X.T)
        elif self.RowColOrder == 'F':
            result = np.kron(Id, X) - np.kron(X.T, Id)
        else:
            raise ValueError("Invalid RowColOrder. Choose 'C' or 'F'.")

        return sparse.csc_matrix(result) if self.SparseM else result

    def AntiCommutationSuperoperator(self, XQ):
        """
        Construct the anti-commutation superoperator {X, •}.

        Parameters
        ----------
        XQ : ndarray or object with .data
            Operator matrix.

        Returns
        -------
        ndarray or sparse matrix
            Superoperator matrix.
        """
        X = np.array(XQ.data) if hasattr(XQ, 'data') else np.array(XQ)
        Id = np.identity(X.shape[-1])

        if self.RowColOrder == 'C':
            result = np.kron(X, Id) + np.kron(Id, X.T)
        elif self.RowColOrder == 'F':
            result = np.kron(Id, X) + np.kron(X.T, Id)
        else:
            raise ValueError("Invalid RowColOrder. Choose 'C' or 'F'.")

        return sparse.csc_matrix(result) if self.SparseM else result

    def Left_Superoperator(self, XQ):
        """
        Construct the left multiplication superoperator: X ⊗ I or I ⊗ X.

        Parameters
        ----------
        XQ : ndarray or object with .data
            Operator matrix.

        Returns
        -------
        ndarray or sparse matrix
            Left superoperator matrix.
        """
        X = np.array(XQ.data) if hasattr(XQ, 'data') else np.array(XQ)
        Id = np.identity(X.shape[-1])

        if self.RowColOrder == 'C':
            result = np.kron(X, Id)
        elif self.RowColOrder == 'F':
            result = np.kron(Id, X)
        else:
            raise ValueError("Invalid RowColOrder. Choose 'C' or 'F'.")

        return sparse.csc_matrix(result) if self.SparseM else result

    def Right_Superoperator(self, XQ):
        """
        Construct the right multiplication superoperator: I ⊗ Xᵀ or Xᵀ ⊗ I.

        Parameters
        ----------
        XQ : ndarray or object with .data
            Operator matrix.

        Returns
        -------
        ndarray or sparse matrix
            Right superoperator matrix.
        """
        X = np.array(XQ.data) if hasattr(XQ, 'data') else np.array(XQ)
        Id = np.identity(X.shape[-1])

        if self.RowColOrder == 'C':
            result = np.kron(Id, X.T)
        elif self.RowColOrder == 'F':
            result = np.kron(X.T, Id)
        else:
            raise ValueError("Invalid RowColOrder. Choose 'C' or 'F'.")

        return sparse.csc_matrix(result) if self.SparseM else result

    def DoubleCommutationSuperoperator(self, XQ, YQ):
        """
        Construct the double commutation superoperator: [X, [Y, •]].

        Parameters
        ----------
        XQ, YQ : ndarray or object with .data
            Operator matrices.

        Returns
        -------
        ndarray or sparse matrix
            Superoperator representing [X, [Y, ρ]].
        """
        X = np.array(XQ.data) if hasattr(XQ, 'data') else np.array(XQ)
        Y = np.array(YQ.data) if hasattr(YQ, 'data') else np.array(YQ)

        Idx = np.identity(X.shape[-1])
        Idy = np.identity(Y.shape[-1])

        if self.RowColOrder == 'C':
            comm_X = np.kron(X, Idx) - np.kron(Idx, X.T)
            comm_Y = np.kron(Y, Idy) - np.kron(Idy, Y.T)
        elif self.RowColOrder == 'F':
            comm_X = np.kron(Idx, X) - np.kron(X.T, Idx)
            comm_Y = np.kron(Idy, Y) - np.kron(Y.T, Idy)
        else:
            raise ValueError("Invalid RowColOrder. Choose 'C' or 'F'.")

        result = np.matmul(comm_X, comm_Y)
        return sparse.csc_matrix(result) if self.SparseM else result
