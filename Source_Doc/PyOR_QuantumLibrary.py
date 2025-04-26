"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module defines the `QuantumLibrary` class, which provides a collection 
    of quantum operations and tools for quantum mechanical systems.

    The `QuantumLibrary` class includes methods for basis construction, tensor operations, 
    commutator evaluation, eigenvalue analysis, partial trace computations, and more, 
    supporting simulations in magnetic resonance and general quantum mechanics.
"""


# ========== Imports ========== #

try:
    from .PyOR_QuantumObject import QunObj
except ImportError:
    from PyOR_QuantumObject import QunObj


import numpy as np
from functools import reduce
import scipy.linalg as la

# ========== Class ========== #

class QuantumLibrary:
    def __init__(self, class_QS=None):
        """
        Initialize the QuantumLibrary with optional quantum system context.

        Parameters
        ----------
        class_QS : object, optional
            Quantum system containing configuration like Sdim, hbarEQ1, etc.
        """
        if class_QS is not None:
            self.class_QS = class_QS
            self.SparseM = class_QS.SparseM
            self.RowColOrder = class_QS.RowColOrder
            self.hbarEQ1 = class_QS.hbarEQ1
        else:
            self.SparseM = False
            self.RowColOrder = 'C'
            self.hbarEQ1 = True

    def Basis_Ket(self, dim, index, PrintDefault=False):
        """
        Generate a ket basis vector `|i⟩` of dimension `dim`.

        Parameters
        ----------
        dim : int
            Dimension of Hilbert space.
        index : int
            Index of the basis vector to set to 1.
        PrintDefault : bool, optional
            Whether to print matrix type info (default is False).

        Returns
        -------
        QunObj
            Ket vector as a QunObj.
        """
        if index < 0 or index >= dim:
            raise ValueError(f"Index must be between 0 and {dim - 1}.")
        state = np.zeros((dim, 1), dtype=complex)
        state[index] = 1.0
        return QunObj(state, PrintDefault=PrintDefault)

    def Basis_Bra(self, dim, index, PrintDefault=False):
        """
        Generate a bra basis vector ⟨i| of dimension `dim`.

        Parameters
        ----------
        dim : int
            Dimension of Hilbert space.
        index : int
            Index of the basis vector to set to 1.
        PrintDefault : bool, optional
            Whether to print matrix type info (default is False).

        Returns
        -------
        QunObj
            Bra vector as a QunObj.
        """
        if index < 0 or index >= dim:
            raise ValueError(f"Index must be between 0 and {dim - 1}.")
        state = np.zeros((1, dim), dtype=complex)
        state[0, index] = 1.0
        return QunObj(state, PrintDefault=PrintDefault)

    def Bloch_Vector(self, theta, phi):
        """
        Generate a qubit state vector `|ψ⟩` on the Bloch sphere.

        Parameters
        ----------
        theta : float
            Polar angle in degrees.
        phi : float
            Azimuthal angle in degrees.

        Returns
        -------
        QunObj
            Bloch state vector as a QunObj.
        """
        vec1 = self.Basis_Ket(2, 0)
        vec2 = self.Basis_Ket(2, 1)
        theta = np.radians(theta)
        phi = np.radians(phi)
        return np.cos(theta / 2) * vec1 + np.exp(1j * phi) * np.sin(theta / 2) * vec2

    def SSpinOp(self, X, String, PrintDefault=False):
        """
        Construct single spin operators (Sx, Sy, Sz, S+, S-) for a given spin S.

        Parameters
        ----------
        X : float
            Spin quantum number (e.g., 1/2, 1, 3/2).
        String : str
            Operator type: 'x', 'y', 'z', 'p', or 'm'.
        PrintDefault : bool, optional
            Whether to print matrix type info.

        Returns
        -------
        QunObj
            Requested spin operator matrix.
        """
        hbar = 1 if self.hbarEQ1 else 1.054e-34
        ms = np.arange(X, -X - 1, -1)
        n = len(ms)

        Sx = np.zeros((n, n), dtype=np.csingle)
        Sy = np.zeros((n, n), dtype=np.csingle)
        Sz = np.zeros((n, n), dtype=np.csingle)
        Sp = np.zeros((n, n), dtype=np.csingle)
        Sm = np.zeros((n, n), dtype=np.csingle)

        Id = np.eye(n)
        Idp = np.triu(np.roll(Id, 1, axis=1), k=1)
        Idn = np.tril(np.roll(Id, -1, axis=1), k=1)

        for i in range(n):
            for j in range(n):
                Sz[i, j] = hbar * ms[j] * Id[i, j]
                Sp[i, j] = np.sqrt(X * (X + 1) - ms[j] * (ms[j] + 1)) * Idp[i, j]
                Sm[i, j] = np.sqrt(X * (X + 1) - ms[j] * (ms[j] - 1)) * Idn[i, j]

        Sx = hbar * 0.5 * (Sp + Sm)
        Sy = hbar * -0.5j * (Sp - Sm)

        if String == 'x':
            return QunObj(Sx, PrintDefault=PrintDefault)
        elif String == 'y':
            return QunObj(Sy, PrintDefault=PrintDefault)
        elif String == 'z':
            return QunObj(Sz, PrintDefault=PrintDefault)
        elif String == 'p':
            return QunObj(Sp, PrintDefault=PrintDefault)
        elif String == 'm':
            return QunObj(Sm, PrintDefault=PrintDefault)
        else:
            raise ValueError("Operator type must be one of 'x', 'y', 'z', 'p', or 'm'.")

    def Purity(self, A: QunObj) -> QunObj:
        """
        Calculate the purity of a quantum state: Tr(ρ²).

        Parameters
        ----------
        A : QunObj
            Density matrix.

        Returns
        -------
        QunObj
            Scalar result of the purity calculation.
        """
        if not isinstance(A, QunObj):
            raise TypeError("Purity only supports QunObj instances.")
        data = np.matmul(A.data, A.data)
        return QunObj(np.trace(data))

    def TensorProduct(self, A: QunObj, B: QunObj) -> QunObj:
        """
        Compute the Kronecker product (tensor product) of two QunObj instances.

        Parameters
        ----------
        A, B : QunObj
            Quantum operators or states.

        Returns
        -------
        QunObj
            Tensor product result.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Tensor product only supports QunObj instances.")
        return QunObj(np.kron(A.data, B.data))

    def TensorProductMultiple(self, *matrices: QunObj) -> QunObj:
        r"""
        Compute tensor product of multiple QunObj instances.

        Parameters
        ----------
        \*matrices : QunObj
            Arbitrary number of quantum operators or states.

        Returns
        -------
        QunObj
            Tensor product of all matrices.
        """
        if not matrices:
            raise ValueError("At least one QunObj must be provided.")
        if not all(isinstance(mat, QunObj) for mat in matrices):
            raise TypeError("Tensor product only supports QunObj instances.")
        result = reduce(np.kron, (m.data for m in matrices))
        return QunObj(result)

    def DirectSum(self, A: QunObj, B: QunObj) -> QunObj:
        """
        Compute the direct sum of two QunObj instances.

        Parameters
        ----------
        A, B : QunObj
            Quantum operators or states.

        Returns
        -------
        QunObj
            Direct sum matrix or ket.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Direct sum only supports QunObj instances.")

        if A.type == "ket" and B.type == "ket":
            return QunObj(np.vstack((A.data, B.data)), Type="ket")

        if A.type == "operator" and B.type == "operator":
            rA, cA = A.data.shape
            rB, cB = B.data.shape
            result = np.zeros((rA + rB, cA + cB), dtype=A.data.dtype)
            result[:rA, :cA] = A.data
            result[rA:, cA:] = B.data
            return QunObj(result, Type="operator")

        raise ValueError("Direct sum supports either two operators or two kets.")

    def DirectSumMultiple(self, *matrices: QunObj) -> QunObj:
        r"""
        Compute direct sum of multiple quantum operators or kets.

        Parameters
        ----------
        \*matrices : QunObj
            Arbitrary number of QunObj instances.

        Returns
        -------
        QunObj
            Resulting direct sum.
        """
        if not matrices:
            raise ValueError("At least one QunObj must be provided.")
        if not all(isinstance(mat, QunObj) for mat in matrices):
            raise TypeError("Direct sum only supports QunObj instances.")

        types = {mat.type for mat in matrices}
        if len(types) > 1:
            raise ValueError("Cannot mix operators and kets.")

        if "ket" in types:
            return QunObj(np.vstack([mat.data for mat in matrices]), Type="ket")

        if "operator" in types:
            total_rows = sum(mat.data.shape[0] for mat in matrices)
            total_cols = sum(mat.data.shape[1] for mat in matrices)
            result = np.zeros((total_rows, total_cols), dtype=matrices[0].data.dtype)

            r, c = 0, 0
            for mat in matrices:
                rows, cols = mat.data.shape
                result[r:r+rows, c:c+cols] = mat.data
                r += rows
                c += cols
            return QunObj(result, Type="operator")

        raise ValueError("Unsupported matrix type for direct sum.")

    def OuterProduct(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the outer product of two QunObj instances.

        Parameters
        ----------
        A : QunObj
            Ket vector.
        B : QunObj
            Bra vector.

        Returns
        -------
        QunObj
            Outer product A ⊗ B† as operator.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Outer product only supports QunObj instances.")
        outer_product = np.outer(A.data, B.data.conj())
        return QunObj(outer_product)

    def InnerProduct(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the inner product ⟨A|B⟩.

        Parameters
        ----------
        A : QunObj
        B : QunObj

        Returns
        -------
        QunObj
            Scalar inner product as a QunObj.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Inner product only supports QunObj instances.")
        return np.matmul(A.data.conj().T, B.data)

    def Identity(self, dim):
        """
        Create an identity matrix.

        Parameters
        ----------
        dim : int
            Dimension.

        Returns
        -------
        QunObj
            Identity matrix.
        """
        return QunObj(np.eye(dim))

    def Zeros(self, dim):
        """
        Create a zero matrix.

        Parameters
        ----------
        dim : int
            Dimension.

        Returns
        -------
        QunObj
            Zero matrix.
        """
        return QunObj(np.zeros((dim, dim)))

    def Commutator(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the commutator [A, B] = AB - BA.

        Parameters
        ----------
        A : QunObj
        B : QunObj

        Returns
        -------
        QunObj
            Commutator matrix.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("Commutator only supports QunObj instances.")
        return QunObj(np.matmul(A.data, B.data) - np.matmul(B.data, A.data))

    def DoubleCommutator(self, A: 'QunObj', B: 'QunObj', rho: 'QunObj') -> 'QunObj':
        """
        Compute the double commutator [A, [B, ρ]].

        Parameters
        ----------
        A, B, rho : QunObj

        Returns
        -------
        QunObj
            Result of [[A, [B, rho]]].
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj) or not isinstance(rho, QunObj):
            raise TypeError("DoubleCommutator only supports QunObj instances.")
        dcommut = np.matmul(B.data, rho.data) - np.matmul(rho.data, B.data)
        commut = np.matmul(A.data, dcommut) - np.matmul(dcommut, A.data)
        return QunObj(commut)

    def AntiCommutator(self, A: 'QunObj', B: 'QunObj') -> 'QunObj':
        """
        Compute the anticommutator {A, B} = AB + BA.

        Parameters
        ----------
        A, B : QunObj

        Returns
        -------
        QunObj
            Anticommutator matrix.
        """
        if not isinstance(A, QunObj) or not isinstance(B, QunObj):
            raise TypeError("AntiCommutator only supports QunObj instances.")
        return QunObj(np.matmul(A.data, B.data) + np.matmul(B.data, A.data))

    def DMToVec(self, rho: 'QunObj', order='C') -> 'QunObj':
        """
        Flatten density matrix into a vector (Liouville representation).

        Parameters
        ----------
        rho : QunObj
        order : str
            'C' (row-major) or 'F' (column-major).

        Returns
        -------
        QunObj
            Vectorized density matrix.
        """
        if not isinstance(rho, QunObj):
            raise TypeError("rho only supports QunObj instances.")
        return QunObj(rho.data.flatten(order).reshape(-1, 1))

    def VecToDM(self, vec: 'QunObj', shape: tuple, order='C') -> 'QunObj':
        """
        Reshape vector back into a matrix.

        Parameters
        ----------
        vec : QunObj
        shape : tuple
            Target matrix shape.
        order : str
            'C' (row-major) or 'F' (column-major).

        Returns
        -------
        QunObj
            Reshaped density matrix.
        """
        if not isinstance(vec, QunObj):
            raise TypeError("vec only supports QunObj instances.")
        return QunObj(vec.data.reshape(shape, order=order))

    def Eigen(self, A: QunObj):
        """
        Compute eigenvalues and eigenvectors of an operator.

        Parameters
        ----------
        A : QunObj

        Returns
        -------
        QunObj, QunObj
            Eigenvalues and eigenvectors.
        """
        if not isinstance(A, QunObj):
            raise TypeError("Eigen only supports QunObj instances.")
        eigenvalues, eigenvectors = np.linalg.eig(A.data)
        return QunObj(eigenvalues.reshape(1, -1).real), QunObj(eigenvectors)

    def Eigen_Split(self, A: QunObj):
        """
        Compute eigenvalues and return each eigenvector as a separate QunObj.

        Parameters
        ----------
        A : QunObj

        Returns
        -------
        QunObj, list of QunObj
            Eigenvalues, list of eigenvectors.
        """
        if not isinstance(A, QunObj):
            raise TypeError("Eigen_Split only supports QunObj instances.")
        eigenvalues, eigenvectors = np.linalg.eig(A.data)
        eigenvalue_obj = QunObj(eigenvalues.reshape(1, -1).real)
        eigenvector_objs = [QunObj(vec.reshape(-1, 1)) for vec in eigenvectors.T]
        return eigenvalue_obj, eigenvector_objs

    def PartialTrace(self, rho: 'QunObj', keep, Sdim = None)-> 'QunObj':
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
        if not isinstance(rho, QunObj):
            raise TypeError("rho must be a QunObj instance.")

        if Sdim is None:
            Sdim = self.class_QS.Sdim.tolist()

        SysInx = range(len(Sdim))
        TraceInx = list(set(SysInx) - set(keep))

        new_shape = Sdim + Sdim
        rho_new = rho.data.reshape(new_shape)

        for idx in sorted(TraceInx, reverse=True):
            rho_new = np.trace(rho_new, axis1=idx, axis2=idx + len(Sdim))

        Sdim_new = [Sdim[j] for j in keep]
        final_shape = (np.prod(Sdim_new), np.prod(Sdim_new))

        return QunObj(rho_new.reshape(final_shape))

    def BlockExtract(self, Matrix: 'QunObj', block_index, block_sizes) -> 'QunObj':
        """
        Extract a specific block from a block-diagonal matrix or ket.

        Parameters
        ----------
        Matrix : QunObj
            Input QunObj instance (operator or ket).
        block_index : int
            Index of the block to extract.
        block_sizes : list of tuple
            List of (row, col) sizes of each block.

        Returns
        -------
        QunObj
            Extracted block as operator or ket.
        """
        if not isinstance(Matrix, QunObj):
            raise TypeError("Matrix must be a QunObj instance.")

        matrix = Matrix.data
        if Matrix.type == "operator":
            start_row = sum(block_sizes[i][0] for i in range(block_index))
            end_row = start_row + block_sizes[block_index][0]
            start_col = sum(block_sizes[i][1] for i in range(block_index))
            end_col = start_col + block_sizes[block_index][1]
            return QunObj(matrix[start_row:end_row, start_col:end_col], Type="operator")

        elif Matrix.type == "ket":
            flat_matrix = matrix.flatten()
            start_index = sum(block_sizes[i][0] for i in range(block_index))
            end_index = start_index + block_sizes[block_index][0]
            reshaped_block = np.reshape(flat_matrix[start_index:end_index], (-1, 1))
            return QunObj(reshaped_block, Type="ket")

        else:
            raise ValueError(f"Unsupported matrix type: {Matrix.type}")
        

    def DMToVec(self, rho: 'QunObj') -> 'QunObj':
        """
        Convert a density matrix to a column vector (vectorization) in Liouville space.

        Parameters
        ----------
        rho : QunObj
            Density matrix.

        Returns
        -------
        QunObj
            Vectorized form of the density matrix.
        """
        if not isinstance(rho, QunObj):
            raise TypeError("rho must be a QunObj instance.")

        if self.RowColOrder == 'C':
            return QunObj(rho.data.flatten('C').reshape(-1, 1))  # Row-major order
        elif self.RowColOrder == 'F':
            return QunObj(rho.data.flatten('F').reshape(-1, 1))  # Column-major order
        else:
            raise ValueError("RowColOrder must be 'C' or 'F'.")

    def VecToDM(self, vec: 'QunObj', shape: tuple) -> 'QunObj':
        """
        Convert a vectorized density matrix back into matrix form.

        Parameters
        ----------
        vec : QunObj
            Vectorized density matrix.
        shape : tuple
            Shape (rows, cols) of the original density matrix.

        Returns
        -------
        QunObj
            Reshaped density matrix.
        """
        if not isinstance(vec, QunObj):
            raise TypeError("vec must be a QunObj instance.")

        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("shape must be a tuple of length 2.")

        if self.RowColOrder == 'C':
            return QunObj(vec.data.reshape(shape, order='C'))
        elif self.RowColOrder == 'F':
            return QunObj(vec.data.reshape(shape, order='F'))
        else:
            raise ValueError("RowColOrder must be 'C' or 'F'.")

    def CommutationSuperoperator(self, X: 'QunObj') -> 'QunObj':
        """
        Create the commutation superoperator [X, ⋅] as a matrix acting on vectorized density matrices.

        Parameters
        ----------
        X : QunObj
            Operator to construct commutator superoperator from.

        Returns
        -------
        QunObj
            Commutation superoperator.
        """
        if not isinstance(X, QunObj):
            raise TypeError("X must be a QunObj instance.")

        I = np.eye(X.shape[-1])
        if self.RowColOrder == 'C':
            L = np.kron(X.data, I) - np.kron(I, X.data.T)
        else:
            L = np.kron(I, X.data) - np.kron(X.data.T, I)

        return QunObj(L)

    def AntiCommutationSuperoperator(self, X: 'QunObj') -> 'QunObj':
        """
        Create the anti-commutation superoperator {X, ⋅} as a matrix acting on vectorized density matrices.

        Parameters
        ----------
        X : QunObj
            Operator to construct anti-commutator superoperator from.

        Returns
        -------
        QunObj
            Anti-commutation superoperator.
        """
        if not isinstance(X, QunObj):
            raise TypeError("X must be a QunObj instance.")

        I = np.eye(X.shape[-1])
        if self.RowColOrder == 'C':
            L = np.kron(X.data, I) + np.kron(I, X.data.T)
        else:
            L = np.kron(I, X.data) + np.kron(X.data.T, I)

        return QunObj(L)

    def DoubleCommutationSuperoperator(self, X: 'QunObj', Y: 'QunObj') -> 'QunObj':
        """
        Construct the double commutator superoperator [[X, ⋅], Y].

        Parameters
        ----------
        X : QunObj
            First operator.
        Y : QunObj
            Second operator.

        Returns
        -------
        QunObj
            Superoperator representing [[X, ⋅], Y].
        """
        if not isinstance(X, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("Inputs must be QunObj instances.")

        I_X = np.eye(X.shape[-1])
        I_Y = np.eye(Y.shape[-1])

        if self.RowColOrder == 'C':
            L_X = np.kron(X.data, I_X) - np.kron(I_X, X.data.T)
            L_Y = np.kron(Y.data, I_Y) - np.kron(I_Y, Y.data.T)
        else:
            L_X = np.kron(I_X, X.data) - np.kron(X.data.T, I_X)
            L_Y = np.kron(I_Y, Y.data) - np.kron(Y.data.T, I_Y)

        return QunObj(np.matmul(L_X, L_Y))

    def Bracket(self, X: 'QunObj', A: 'QunObj', Y: 'QunObj') -> 'QunObj':
        """
        Compute the scalar bracket ⟨X|A|Y⟩ = Tr(X† A Y)

        Parameters
        ----------
        X : QunObj
            First operator or state (conjugated from the left).
        A : QunObj
            Middle operator.
        Y : QunObj
            Final operator or state.

        Returns
        -------
        complex
            Scalar result of the bracket expression.
        """
        if not isinstance(X, QunObj) or not isinstance(A, QunObj) or not isinstance(Y, QunObj):
            raise TypeError("All inputs must be instances of QunObj.")

        return np.trace(np.matmul(X.data.conj().T, np.matmul(A.data, Y.data)))

    def Eigen(self, A: QunObj) -> tuple:
        """
        Compute eigenvalues and eigenvectors of a quantum operator.

        Parameters
        ----------
        A : QunObj
            Operator to decompose.

        Returns
        -------
        tuple
            (eigenvalues, eigenvectors) where:
            - eigenvalues: QunObj with shape (1, N) as a row of real parts.
            - eigenvectors: QunObj with shape (N, N) containing eigenvectors column-wise.
        """
        if not isinstance(A, QunObj):
            raise TypeError("Input must be a QunObj instance.")

        eigenvalues, eigenvectors = la.eig(A.data)
        return QunObj(eigenvalues.reshape(1, -1).real), QunObj(eigenvectors)

    def Eigen_Split(self, A: QunObj) -> tuple:
        """
        Compute eigenvalues and eigenvectors of a quantum operator.
        Eigenvectors are returned as separate QunObj columns.

        Parameters
        ----------
        A : QunObj
            Operator to decompose.

        Returns
        -------
        tuple
            (eigenvalues, eigenvector_list)
            - eigenvalues: QunObj with shape (1, N)
            - eigenvector_list: list of QunObj, each column vector (N x 1)
        """
        if not isinstance(A, QunObj):
            raise TypeError("Input must be a QunObj instance.")

        eigenvalues, eigenvectors = la.eig(A.data)
        eigenvalue_obj = QunObj(eigenvalues.reshape(1, -1).real)
        eigenvector_objs = [QunObj(vec.reshape(-1, 1)) for vec in eigenvectors.T]

        return eigenvalue_obj, eigenvector_objs

