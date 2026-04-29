"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This module provides functions related to rotation operations in quantum mechanics 
    and magnetic resonance simulations.

    Functions include generating rotation matrices, applying rotations to quantum states 
    and operators, and supporting Euler angle-based transformations.
"""

import numpy as np
from scipy.linalg import expm

try:
    from .PyOR_QuantumObject import QunObj
    from .PyOR_QuantumLibrary import QuantumLibrary
except ImportError:
    from PyOR_QuantumObject import QunObj
    from PyOR_QuantumLibrary import QuantumLibrary

QLib = QuantumLibrary()

def RotateX(theta):
    r"""
    Rotates a vector or tensor about the X-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the X-axis in three-dimensional space. The angle 
    :math:`\theta` is provided in degrees and is internally converted to radians.

    Parameters
    ----------
    theta : float
        The angle of rotation in degrees. The function converts this to radians 
        for the rotation calculation.

    Returns
    -------
    numpy.ndarray
        A 3x3 NumPy array representing the rotation matrix for a counterclockwise 
        rotation about the X-axis by the angle :math:`\theta`.

    Notes
    -----
    The rotation matrix for a counterclockwise rotation about the X-axis is:

    .. math::

        \begin{bmatrix}
        1 & 0          & 0 \\
        0 & \cos\theta & -\sin\theta \\
        0 & \sin\theta & \cos\theta
        \end{bmatrix}

    where :math:`\theta` is the angle of rotation in radians.
    """


    theta = theta * np.pi / 180.0
    return np.asarray([[1,0,0],[0, np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])

def RotateY(theta):
    r"""
    Rotates a vector or tensor about the Y-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the Y-axis in three-dimensional space. The angle 
    :math:`\theta` is provided in degrees and is internally converted to radians.

    Parameters
    ----------
    theta : float
        The angle of rotation in degrees. The function converts this to radians 
        for the rotation calculation.

    Returns
    -------
    numpy.ndarray
        A 3x3 NumPy array representing the rotation matrix for a counterclockwise 
        rotation about the Y-axis by the angle :math:`\theta`.

    Notes
    -----
    The rotation matrix for a counterclockwise rotation about the Y-axis is:

    .. math::

        \begin{bmatrix}
        \cos\theta & 0 & \sin\theta \\
        0          & 1 & 0 \\
        -\sin\theta & 0 & \cos\theta
        \end{bmatrix}

    where :math:`\theta` is the angle of rotation in radians.
    """

    theta = theta * np.pi / 180.0
    return np.asarray([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    
def RotateZ(theta):
    r"""
    Rotates a vector or tensor about the Z-axis by a given angle.

    This function returns a 3x3 rotation matrix that can be applied to a vector or tensor
    to perform a rotation about the Z-axis in three-dimensional space. The angle :math:`\theta` is
    provided in degrees and is internally converted to radians.

    Parameters
    ----------
    theta : float
        The angle of rotation in degrees. The function converts this to radians 
        for the rotation calculation.

    Returns
    -------
    numpy.ndarray
        A 3x3 NumPy array representing the rotation matrix for a counterclockwise rotation 
        about the Z-axis by the angle :math:`\theta`.

    Notes
    -----
    The rotation matrix for a counterclockwise rotation about the Z-axis is:

    .. math::

        \begin{bmatrix}
        \cos\theta & -\sin\theta & 0 \\
        \sin\theta &  \cos\theta & 0 \\
        0          &  0          & 1
        \end{bmatrix}

    where :math:`\theta` is the angle of rotation in radians.
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
    r"""
    Computes the Wigner d-matrix for a given rank and angle.

    The Wigner d-matrix is used to represent the rotation of spherical harmonics
    or quantum states under rotations in quantum mechanics. It is often used in 
    various fields, including nuclear magnetic resonance (NMR) and quantum chemistry.
    This function computes the Wigner d-matrix for a given rank (angular momentum quantum number)
    and a rotation angle :math:`\beta` (in degrees).

    Parameters
    ----------
    rank : int
        The rank (or angular momentum quantum number) for which the Wigner d-matrix is computed.
        This corresponds to the quantum number :math:`l` in the Wigner d-matrix formulation.

    beta : float
        The rotation angle :math:`\beta` in degrees. The function converts this to radians 
        for the computation.

    Returns
    -------
    QunObj
        A quantum object wrapping the computed Wigner d-matrix. The matrix is unitary 
        and has shape :math:`(2l + 1) \times (2l + 1)`.

    Notes
    -----
    The Wigner d-matrix :math:`d^l_{m,m'}(\beta)` describes how spherical harmonics or 
    spin states transform under rotations. In this function:

    - :math:`l` is the angular momentum quantum number (rank)
    - :math:`\beta` is the rotation angle around the Y-axis, in degrees
    - The function uses the spin operator :math:`S_y` and applies the exponential:  

    :math:`\exp(-i \, \beta \, S_y)`

    See Also
    --------
    expm : Matrix exponential function.  
    QLib.SSpinOp : Function to generate spin operators.

    References
    ----------
    1. Nouredine Zettili, *Quantum Mechanics: Concepts and Applications*.
    """


    SyQ = QLib.SSpinOp(rank,"y",PrintDefault=False)
    Sy = SyQ.data
    beta = beta * np.pi / 180.0
    return QunObj(expm(-1j * beta * Sy))

def Wigner_D_Matrix(rank,alpha,beta,gamma):
    r"""
    Computes the Wigner D-matrix for a given rank and three Euler angles (alpha, beta, gamma).

    The Wigner D-matrix is used to describe rotations in quantum mechanics and is particularly
    useful in representing the rotation of angular momentum eigenstates. The matrix is constructed
    using the three Euler angles: alpha (:math:`\alpha`), beta (:math:`\beta`), and gamma (:math:`\gamma`), 
    which correspond to rotations around the Z-axis, Y-axis, and Z-axis again, respectively.

    Parameters
    ----------
    rank : int
        The angular momentum quantum number :math:`l` for which the Wigner D-matrix is computed.
        The resulting matrix is of size :math:`(2l + 1) \times (2l + 1)`.

    alpha : float
        Rotation angle around the first Z-axis (:math:`\alpha`) in degrees.

    beta : float
        Rotation angle around the Y-axis (:math:`\beta`) in degrees.

    gamma : float
        Rotation angle around the second Z-axis (:math:`\gamma`) in degrees.

    Returns
    -------
    QunObj
        A quantum object wrapping the computed Wigner D-matrix, which is a unitary matrix 
        of size :math:`(2l + 1) \times (2l + 1)`.

    Notes
    -----
    The Wigner D-matrix for the given Euler angles is computed using the following expression:

    .. math::

        D^l_{m,m'}(\alpha, \beta, \gamma) = 
        \exp(-i \alpha S_z) \cdot \exp(-i \beta S_y) \cdot \exp(-i \gamma S_z)

    where:

    - :math:`S_z` is the spin operator along the Z-axis.  
    - :math:`S_y` is the spin operator along the Y-axis.  
    - The angles :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are the Euler angles in **radians**.

    The Wigner D-matrix represents the rotation of quantum states under the specified Euler angles.

    See Also
    --------
    expm : Matrix exponential function  
    QLib.SSpinOp : Generates spin operators for a given rank

    References
    ----------
    1. Nouredine Zettili, *Quantum Mechanics: Concepts and Applications*.
    """

    SyQ = QLib.SSpinOp(rank,"y",PrintDefault=False)
    SzQ = QLib.SSpinOp(rank,"z",PrintDefault=False)

    Sy = SyQ.data
    Sz = SzQ.data

    alpha = alpha * np.pi / 180.0
    beta = beta * np.pi / 180.0
    gamma = gamma * np.pi / 180.0

    return QunObj(expm(-1j * alpha * Sz) @ expm(-1j * beta * Sy) @ expm(-1j * gamma * Sz))
