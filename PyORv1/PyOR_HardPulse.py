"""
PyOR - Python On Resonance

Author:
    Vineeth Francis Thalakottoor Jose Chacko

Email:
    vineethfrancis.physics@gmail.com

Description:
    This file defines the `HardPulse` class, which implements methods for handling 
    rotation (pulse) operations in both Hilbert and Liouville spaces as applied 
    in magnetic resonance simulations.

    The `HardPulse` class supports the construction of rotation operators, 
    pulse applications, and conversions between different quantum mechanical 
    representations.
"""


import numpy as np
from scipy.linalg import expm

try:
    from .PyOR_Commutators import Commutators as COM
    from .PyOR_QuantumObject import QunObj
except ImportError:
    from PyOR_Commutators import Commutators as COM
    from PyOR_QuantumObject import QunObj


class HardPulse:
    def __init__(self, class_QS):
        """
        Initialize the HardPulse class.

        Parameters
        ----------
        class_QS : object
            Instance of QuantumSystem, providing spin operators and propagation space settings.
        """
        self.class_QS = class_QS
        self.class_COM = COM()

    def Rotate_CyclicPermutation(self, AQ, BS, theta):
        """
        Rotate operator BS about operator AQ using cyclic commutation relations.

        This method applies an analytical rotation based on the cyclic commutator rule:
        If [A, B] = jC, then the rotation of B about A by angle θ is given by:

            EXP(-j A * θ) @ B @ EXP(j A * θ) 
                = B * cos(θ) - j * [A, B] * sin(θ) 
                = B * cos(θ) + C * sin(θ)

        where:
            A : Operator about which rotation is applied
            B : Operator being rotated
            C : Operator resulting from the commutator [A, B] = jC
            j : imaginary unit

        Parameters
        ----------
        AQ : QunObj
            Operator A about which the rotation happens.
        BS : QunObj
            Operator B to be rotated.
        theta : float
            Rotation angle in degrees.

        Returns
        -------
        QunObj
            Rotated operator.
        """
        if not isinstance(AQ, QunObj) or not isinstance(BS, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        A = AQ.data
        B = BS.data

        if np.allclose(A, B):
            Bp = B
        else:
            Bp = B * np.cos(np.pi * theta / 180.0) - 1j * self.class_COM.Commutator(A, B) * np.sin(np.pi * theta / 180.0)

        return QunObj(Bp)

    def Rotate_Pulse(self, rhoQ, theta_rad, operatorQ):
        """
        Perform rotation on an operator or state under a pulse.

        This applies the unitary rotation: exp(-iθA) ρ exp(iθA)

        Parameters
        ----------
        rhoQ : QunObj
            Density matrix or operator to be rotated.
        theta_rad : float
            Rotation angle in degrees.
        operatorQ : QunObj
            Operator generating the rotation (e.g., Ix, Iy, Iz).

        Returns
        -------
        QunObj
            Rotated operator or state.
        """
        if not isinstance(rhoQ, QunObj) or not isinstance(operatorQ, QunObj):
            raise TypeError("Both inputs must be instances of QunObj.")

        rho = rhoQ.data
        operator = operatorQ.data
        theta_rad = np.pi * theta_rad / 180.0  # Convert degrees to radians

        if self.class_QS.PropagationSpace == "Hilbert":
            U = expm(-1j * theta_rad * operator)
            rotated = U @ rho @ U.T.conj()
        elif self.class_QS.PropagationSpace == "Liouville":
            L = self.class_COM.CommutationSuperoperator(self.class_QS.Class_quantumlibrary.VecToDM(QunObj(operator),(self.class_QS.Vdim,self.class_QS.Vdim)).data)
            rotated = expm(-1j * theta_rad * L) @ rho
        else:
            raise ValueError("Unknown propagation space specified in class_QS.")

        return QunObj(rotated)
