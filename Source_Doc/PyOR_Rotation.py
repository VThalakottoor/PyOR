"""
PyOR Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
email: vineethfrancis.physics@gmail.com

This file contain functions related to Rotation

Documentation is done.
"""

import numpy as np
from scipy.linalg import expm

from PyOR_QuantumObject import QunObj
from PyOR_QuantumLibrary import QuantumLibrary
QLib = QuantumLibrary()

def RotateX(theta):
    """
    Rotates a vector or tensor about the X-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the X-axis in three-dimensional space. The angle `theta` is
    provided in degrees and is internally converted to radians.

    Parameters:
    -----------
    theta : float
        The angle of rotation in degrees. The function will convert this to radians for 
        the rotation calculation.

    Returns:
    --------
    numpy.ndarray
        A 3x3 numpy array representing the rotation matrix for a counterclockwise rotation 
        about the X-axis by the angle `theta`.

    Notes:
    ------
    The rotation matrix for a counterclockwise rotation about the X-axis is given by:

        | 1      0         0    |
        | 0   cos(θ)   -sin(θ) |
        | 0   sin(θ)    cos(θ) |

    where θ is the angle of rotation in radians.
    """

    theta = theta * np.pi / 180.0
    return np.asarray([[1,0,0],[0, np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def RotateY(theta):
    """
    Rotates a vector or tensor about the Y-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the Y-axis in three-dimensional space. The angle `theta` is
    provided in degrees and is internally converted to radians.

    Parameters:
    -----------
    theta : float
        The angle of rotation in degrees. The function will convert this to radians for 
        the rotation calculation.

    Returns:
    --------
    numpy.ndarray
        A 3x3 numpy array representing the rotation matrix for a counterclockwise rotation 
        about the Y-axis by the angle `theta`.

    Notes:
    ------
    The rotation matrix for a counterclockwise rotation about the Y-axis is given by:

        | cos(θ)    0    sin(θ) |
        |    0      1       0   |
        | -sin(θ)   0    cos(θ) |

    where θ is the angle of rotation in radians.
    """
    theta = theta * np.pi / 180.0
    return np.asarray([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    
def RotateZ(theta):
    """
    Rotates a vector or tensor about the Z-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the Z-axis in three-dimensional space. The angle `theta` is
    provided in degrees and is internally converted to radians.

    Parameters:
    -----------
    theta : float
        The angle of rotation in degrees. The function will convert this to radians for 
        the rotation calculation.

    Returns:
    --------
    numpy.ndarray
        A 3x3 numpy array representing the rotation matrix for a counterclockwise rotation 
        about the Z-axis by the angle `theta`.

    Notes:
    ------
    The rotation matrix for a counterclockwise rotation about the Z-axis is given by:

        | cos(θ)  -sin(θ)  0 |
        | sin(θ)   cos(θ)  0 |
        |   0        0     1 |

    where θ is the angle of rotation in radians.
    """
    theta = theta * np.pi / 180.0
    return np.asarray([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def RotateEuler(alpha,beta,gamma):
    """
    Performs a rotation using Euler angles (α, β, γ) in three-dimensional space.

    This function computes the overall rotation matrix by applying a series of three
    rotations around the Z-axis, Y-axis, and Z-axis again, using the provided Euler angles:
    - alpha (α) : rotation about the Z-axis
    - beta (β)  : rotation about the Y-axis
    - gamma (γ) : rotation about the Z-axis

    The rotation order follows the intrinsic Tait-Bryan angle convention (Z-Y-Z), meaning:
    1. Rotate by α around the Z-axis.
    2. Rotate by β around the Y-axis.
    3. Rotate by γ around the Z-axis again.

    Parameters:
    -----------
    alpha : float
        The angle of rotation about the Z-axis (α) in degrees.
    beta : float
        The angle of rotation about the Y-axis (β) in degrees.
    gamma : float
        The angle of rotation about the Z-axis (γ) in degrees.

    Returns:
    --------
    numpy.ndarray
        A 3x3 numpy array representing the resulting rotation matrix for the combined Euler rotations.
    
    Notes:
    ------
    The combined rotation is calculated as:

        R = RotateZ(α) * RotateY(β) * RotateZ(γ)

    where:
        - RotateZ(α) applies a counterclockwise rotation by α around the Z-axis.
        - RotateY(β) applies a counterclockwise rotation by β around the Y-axis.
        - RotateZ(γ) applies a counterclockwise rotation by γ around the Z-axis.
    """
    return RotateZ(alpha) @ RotateY(beta) @ RotateZ(gamma)


def Wigner_d_Matrix(rank, beta):
    """
    Computes the Wigner d-matrix for a given rank and angle.

    The Wigner d-matrix is used to represent the rotation of spherical harmonics
    or quantum states under rotations in quantum mechanics. It is often used in 
    various fields, including nuclear magnetic resonance (NMR) and quantum chemistry.
    This function computes the Wigner d-matrix for a given rank (angular momentum quantum number)
    and a rotation angle `beta` (in degrees).

    Parameters:
    -----------
    rank : int
        The rank (or angular momentum quantum number) for which the Wigner d-matrix is computed.
        This corresponds to the quantum number `l` in the Wigner d-matrix formulation.
        
    beta : float
        The rotation angle (β) in degrees. The function will convert this angle to radians for 
        the computation of the matrix.

    Returns:
    --------
    QunObj
        A custom object (assumed from context) wrapping the computed Wigner d-matrix, 
        which is represented as a unitary matrix. The matrix is of size (2*rank + 1) x (2*rank + 1).
    
    Notes:
    ------
    The Wigner d-matrix `d^l_{m,m'}(β)` describes how spherical harmonics (or spin states) 
    transform under rotations. In this function:
        - `rank` is the angular momentum quantum number `l`.
        - `beta` is the rotation angle around the Y-axis (typically) in degrees.
        - The function uses the spin operator `Sy` to calculate the matrix and applies 
          the exponential of the operator with `exp(-1j * beta * Sy)`.

    See Also:
    ---------
    expm : Matrix exponential function.
    QLib.SSpinOp : Function to generate spin operators.
    
    Reference:
    ----------
    1. Quantum Mechanics: Concepts and Applications, Nouredine Zettili.
    """

    SyQ = QLib.SSpinOp(rank,"y",PrintDefault=False)
    Sy = SyQ.data
    beta = beta * np.pi / 180.0
    return QunObj(expm(-1j * beta * Sy))

def Wigner_D_Matrix(rank,alpha,beta,gamma):
    """
    Computes the Wigner D-matrix for a given rank and three Euler angles (alpha, beta, gamma).

    The Wigner D-matrix is used to describe rotations in quantum mechanics and is particularly
    useful in representing the rotation of angular momentum eigenstates. The matrix is constructed
    using the three Euler angles: alpha (α), beta (β), and gamma (γ), which correspond to rotations
    around the Z-axis, Y-axis, and Z-axis again, respectively.

    Parameters:
    -----------
    rank : int
        The angular momentum quantum number `l` for which the Wigner D-matrix is computed. The size
        of the resulting matrix is `(2*rank + 1) x (2*rank + 1)`.

    alpha : float
        The angle of rotation around the first Z-axis (α) in degrees. This angle is converted to radians
        for the calculation.

    beta : float
        The angle of rotation around the Y-axis (β) in degrees. This angle is converted to radians
        for the calculation.

    gamma : float
        The angle of rotation around the second Z-axis (γ) in degrees. This angle is converted to radians
        for the calculation.

    Returns:
    --------
    QunObj
        A custom object (assumed from context) wrapping the computed Wigner D-matrix, 
        which is a unitary matrix of size `(2*rank + 1) x (2*rank + 1)`.

    Notes:
    ------
    The Wigner D-matrix for the given Euler angles is computed using the following formula:

        D^l_{m,m'}(α,β,γ) = exp(-i α Sz) * exp(-i β Sy) * exp(-i γ Sz)

    where:
        - `Sz` is the spin operator along the Z-axis.
        - `Sy` is the spin operator along the Y-axis.
        - The angles `alpha`, `beta`, and `gamma` are the Euler angles in radians.

    The Wigner D-matrix is used to represent rotations of quantum states under the specified 
    Euler angles.

    See Also:
    ---------
    expm : Matrix exponential function.
    QLib.SSpinOp : Function to generate spin operators for a given rank.

    Reference:
    ----------
    1. Quantum Mechanics: Concepts and Applications, Nouredine Zettili.
    """
    SyQ = QLib.SSpinOp(rank,"y",PrintDefault=False)
    SzQ = QLib.SSpinOp(rank,"z",PrintDefault=False)

    Sy = SyQ.data
    Sz = SzQ.data

    alpha = alpha * np.pi / 180.0
    beta = beta * np.pi / 180.0
    gamma = gamma * np.pi / 180.0

    return QunObj(expm(-1j * alpha * Sz) @ expm(-1j * beta * Sy) @ expm(-1j * gamma * Sz))
