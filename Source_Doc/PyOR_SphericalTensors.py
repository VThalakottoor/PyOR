"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain functions related to spherical tensors

Documentation is done
"""

import numpy as np
from PyOR_QuantumObject import QunObj

def MatrixToSphericalTensors(AQ):
    """
    Converts a 3x3 Cartesian matrix into its corresponding spherical tensor components.

    This function decomposes a Cartesian second-rank tensor (typically used in NMR or other 
    physics applications) into spherical tensor components of rank 0 (isotropic), rank 1 
    (antisymmetric), and rank 2 (symmetric traceless). The decomposition follows the formalism 
    described in Pascal P. Man's 2014 paper on Cartesian and spherical tensors in NMR Hamiltonians.

    Parameters:
    -----------
    AQ : numpy.matrix or similar object
        A 3x3 matrix (usually Hermitian or real) representing the Cartesian tensor 
        to be converted. AQ must be a quantum object with attribute "data".

    Returns:
    --------
    dict
        A dictionary containing the spherical tensor components:
            - "rank0": complex
                The rank-0 (isotropic) component T(0,0).
            - "rank1": list of complex
                The rank-1 (antisymmetric) components [T(1,1), T(1,0), T(1,-1)].
            - "rank2": list of complex
                The rank-2 (symmetric traceless) components [T(2,2), T(2,1), T(2,0), T(2,-1), T(2,-2)].

    Reference:
    ----------
    1. Pascal P. Man, "Cartesian and Spherical Tensors in NMR Hamiltonians", 
    Concepts in Magnetic Resonance Part A, 2014. https://doi.org/10.1002/cmr.a.21289
    (Equations 275 to 281 are particularly relevant)

    2. Tensors and Rotations in NMR, LEONARD J. MUELLER,  https://doi.org/10.1002/cmr.a.20224
    """
    
    A = AQ.data
    Sptensor = {}

    # Isotropic (rank 0)
    Sptensor["rank0"] = (-1/np.sqrt(3)) * (A[0][0] + A[1][1] + A[2][2]) # T(0,0)

    # Anti-Symmetric (rank 1)
    AANTI_1 = (-1.0/2.0) * (A[2][0] - A[0][2] - 1j *(A[2][1] - A[1][2])) # T(1,-1)
    AANTI_2 = (-1.0/np.sqrt(2)) * 1j * (A[0][1] - A[1][0]) # T(1,0)
    AANTI_3 = (-1.0/2.0) * (A[2][0] - A[0][2] + 1j *(A[2][1] - A[1][2])) # T(1,1)
    Sptensor["rank1"] = [AANTI_3,AANTI_2,AANTI_1]

    # Symmetric (rank 2)
    SYMM_1 = (1.0/2.0) * (A[0][0] - A[1][1] - 1j *(A[0][1] + A[1][0])) # T(2,-2)
    SYMM_2 = (1.0/2.0) * (A[0][2] + A[2][0] - 1j *(A[1][2] + A[2][1])) # T(2,-1)
    SYMM_3 = (1.0/np.sqrt(6)) * (3 * A[2][2] - (A[0][0] + A[1][1] + A[2][2])) # T(2,0)
    SYMM_4 = (-1.0/2.0) * (A[0][2] + A[2][0] + 1j *(A[1][2] + A[2][1])) # T(2,1)
    SYMM_5 = (1.0/2.0) * (A[0][0] - A[1][1] + 1j *(A[0][1] + A[1][0])) # T(2,2)
    Sptensor["rank2"] = [SYMM_5,SYMM_4,SYMM_3,SYMM_2,SYMM_1]

    return Sptensor

def SphericalTensorsToMatrix(Sptensor):
    """
    Reconstructs a 3x3 complex Cartesian matrix from its spherical tensor components.

    This function performs the inverse operation of `MatrixToSphericalTensors`. It takes
    a dictionary containing spherical tensor components of ranks 0 (isotropic), 1 
    (antisymmetric), and 2 (symmetric traceless), and reconstructs the corresponding
    Cartesian second-rank tensor as a 3x3 complex NumPy array.

    Parameters:
    -----------
    Sptensor : dict
        Dictionary containing spherical tensor components with the following keys:
            - 'rank0': complex
                Scalar representing the isotropic component T(0,0).
            - 'rank1': list of 3 complex numbers
                Antisymmetric components [T(1,1), T(1,0), T(1,-1)].
            - 'rank2': list of 5 complex numbers
                Symmetric traceless components [T(2,2), T(2,1), T(2,0), T(2,-1), T(2,-2)].

    Returns:
    --------
    QunObj
        A custom object (assumed from context) wrapping a 3x3 complex NumPy array 
        that represents the reconstructed Cartesian tensor.

    Notes:
    ------
    - The spherical to Cartesian transformation follows standard NMR tensor decomposition
      conventions, particularly those described in:

    - This function assumes `QunObj` is a class or wrapper that accepts a 3x3 complex 
      NumPy array. Ensure `QunObj` is defined in your codebase or environment.

    See Also:
    ---------
    MatrixToSphericalTensors : Function that performs the forward decomposition from a 
                               Cartesian tensor to spherical tensor components.


    Reference:
    ----------
    1. Tensors and Rotations in NMR, LEONARD J. MUELLER,  https://doi.org/10.1002/cmr.a.20224
    """

    T0 = Sptensor["rank0"]
    T11, T10, T1m1 = Sptensor["rank1"]
    T22, T21, T20, T2m1, T2m2 = Sptensor["rank2"]

    # Spherical tensor Basis
    TB00 = (-1/np.sqrt(3) ) * np.eye(3)

    TB10 = (-1 / np.sqrt(2)) * np.array([[0, 1j, 0], [-1j, 0, 0], [0, 0, 0]])
    TB11 = 0.5 * np.array([[0, 0, -1], [0, 0, -1j], [1, 1j, 0]])
    TB1m1 = 0.5 * np.array([[0, 0, -1], [0, 0, 1j], [1, -1j, 0]])

    TB20 = (1.0 / np.sqrt(6)) * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]])
    TB21 = - 0.5 * np.array([[0, 0, 1], [0, 0, 1j], [1, 1j, 0]])
    TB2m1 = 0.5 * np.array([[0, 0, 1], [0, 0, -1j], [1, -1j, 0]])
    TB22 = 0.5  * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]])
    TB2m2 = 0.5 * np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0]])

    A = np.zeros((3, 3), dtype=complex)

    # Rank 0 (isotropic)
    A = A + T0 * TB00

    # Rank 1 (antisymmetric)
    A = A + T1m1 * TB1m1
    A = A + T10 * TB10
    A = A + T11 * TB11

    # Rank 2 (symmetric traceless)
    A = A + T2m2 * TB2m2
    A = A + T2m1 * TB2m1
    A = A + T20 * TB20
    A = A + T21 * TB21
    A = A + T22 * TB22

    return QunObj(A)