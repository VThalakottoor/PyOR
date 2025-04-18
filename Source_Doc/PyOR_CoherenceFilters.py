"""
PyOR - Python On Resonance
Author: Vineeth Francis Thalakottoor Jose Chacko
Email: vineethfrancis.physics@gmail.com

This file contains the class CoherenceFilter.

Documentation is done.
"""

import numpy as np

from PyOR_QuantumObject import QunObj
from PyOR_Basis import Basis
from PyOR_Hamiltonian import Hamiltonian


class CoherenceFilter:
    def __init__(self, class_QS):
        """
        Initialize CoherenceFilter with quantum system state.

        Parameters
        ----------
        class_QS : QuantumSystem
            An instance of the quantum system with spin operators.
        """
        self.class_QS = class_QS
        self.class_Basis = Basis(class_QS)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Coherence Filter
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Filter_T00(self, rho, index):
        """
        T00 Filter: Extracts zero quantum coherence for selected spins.
        Works on the density matrix in the Zeeman basis.

        Parameters
        ----------
        rho : ndarray
            Input density matrix.
        index : list or tuple of int
            Indices of the two spins to apply the filter on.

        Returns
        -------
        tuple (Filter_T00, filtered_rho)
            Filter matrix and the filtered density matrix.
        """
        Sx = self.class_QS.Sx_
        Sy = self.class_QS.Sy_
        Sz = self.class_QS.Sz_

        ZQx = Sx[index[0]] @ Sx[index[1]] + Sy[index[0]] @ Sy[index[1]] + Sz[index[0]] @ Sz[index[1]]
        ZQy = Sy[index[0]] @ Sx[index[1]] - Sx[index[0]] @ Sy[index[1]]

        Filter_T00 = ZQx.copy()
        Filter_T00[np.isin(Filter_T00, [0.5, -0.5, 0.25, -0.25])] = 1

        return Filter_T00, np.multiply(rho, Filter_T00)

    def Filter_Coherence(self, rho, Allow_Coh):
        """
        Filter to allow only selected coherence order.
        Operates on the density matrix in the Zeeman basis.

        Parameters
        ----------
        rho : ndarray
            Input density matrix.
        Allow_Coh : int
            The allowed coherence order to retain.

        Returns
        -------
        tuple (coherence_mask, filtered_rho)
            The coherence mask array and filtered density matrix.
        """
        Sz = self.class_QS.Sz_

        Basis_Zeeman, dic_Zeeman, coh_Zeeman, coh_Zeeman_array = self.class_Basis.ProductOperators_Zeeman()
        Max_Coh = int(np.max(coh_Zeeman_array))

        # Mark allowed coherence values
        coh_Zeeman_array[coh_Zeeman_array == Allow_Coh] = 1000

        # Zero out all others
        for i in range(Max_Coh + 1):
            coh_Zeeman_array[coh_Zeeman_array == i] = 0
            coh_Zeeman_array[coh_Zeeman_array == -i] = 0

        coh_Zeeman_array = coh_Zeeman_array / 1000.0

        return coh_Zeeman_array, np.multiply(rho, coh_Zeeman_array)
