"""
PyOR - Python On Resonance

Multiple quantum-system container and optional direct-sum manifold support.

This module intentionally keeps QuantumSystem in PyOR_QuantumSystem.py
and builds higher-level combinations without modifying individual systems.
"""

import numpy as np
from itertools import product

try:
    from .PyOR_QuantumSystem import QuantumSystem
    from .PyOR_QuantumObject import QunObj
except ImportError:
    from PyOR_QuantumSystem import QuantumSystem
    from PyOR_QuantumObject import QunObj


class _ManifoldView:
    """Namespace for operators embedded in the full manifold space."""
    pass


class QuantumSystems:
    """
    Container for multiple independent QuantumSystem objects.

    By default all contained systems remain fully independent.
    A direct-sum manifold is created only when CreateManifold() is called.
    """

    def __init__(self, *Systems, PrintDefault=False):

        if len(Systems) == 0:
            raise ValueError("At least one quantum system must be provided.")

        self.System = {}

        for system_definition in Systems:

            if not isinstance(system_definition, dict):
                raise TypeError(
                    "Each quantum system must be provided as a dictionary."
                )

            if len(system_definition) != 1:
                raise ValueError(
                    "Each system dictionary must contain exactly one system name."
                )

            name, SpinList = next(iter(system_definition.items()))

            if not isinstance(name, str):
                raise TypeError("Quantum system name must be a string.")

            if not isinstance(SpinList, dict):
                raise TypeError(
                    f"Spin list for system '{name}' must be a dictionary."
                )

            if name in self.System:
                raise ValueError(
                    f"Quantum system '{name}' already exists."
                )

            # Existing PyOR behavior: every entry is a normal QuantumSystem.
            quantum_system = QuantumSystem(
                SpinList,
                PrintDefault=PrintDefault
            )

            self.System[name] = quantum_system
            setattr(self, name, quantum_system)

        self.Nsystems = len(self.System)

        # Manifold support is opt-in.
        self.ManifoldCreated = False
        self.Manifolds = []

    def __getitem__(self, name):
        return self.System[name]

    def __len__(self):
        return self.Nsystems

    def __iter__(self):
        return iter(self.System)

    @property
    def Vdims(self):
        return {
            name: system.Vdim
            for name, system in self.System.items()
        }

    @property
    def Ldims(self):
        return {
            name: system.Ldim
            for name, system in self.System.items()
        }

    @property
    def Nspins_system(self):
        return {
            name: system.Nspins
            for name, system in self.System.items()
        }

    def Configure(self, auto_update=True, **kwargs):
        """
        Configure multiple independent quantum systems.

        Common parameters are applied to every system. Named dictionaries
        provide system-specific overrides. This preserves the existing
        Chemical Exchange behavior.
        """

        common_config = {}
        system_config = {}

        for key, value in kwargs.items():

            if key in self.System:
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Configuration for system '{key}' must be a dictionary."
                    )
                system_config[key] = value
            else:
                common_config[key] = value

        for name, system in self.System.items():

            config = common_config.copy()

            if name in system_config:
                config.update(system_config[name])

            Jcouplings = config.pop("Jcouplings", None)

            system.Configure(
                Jcouplings=Jcouplings,
                auto_update=auto_update,
                **config
            )

        return self

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Direct-sum manifold support
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def CreateManifold(self, Manifolds=None, QuantumObject=True):
        """
        Create a direct-sum Hilbert space from selected QuantumSystem objects.

        The individual systems themselves are not modified.

        Parameters
        ----------
        Manifolds : sequence of str, optional
            System names to combine. If None, all systems are used.
        QuantumObject : bool, optional
            Store manifold projectors and identity as QunObj when True.

        Returns
        -------
        QuantumSystems
            This container with manifold bookkeeping added.
        """

        if Manifolds is None:
            Manifolds = list(self.System.keys())
        else:
            Manifolds = list(Manifolds)

        if len(Manifolds) == 0:
            raise ValueError(
                "At least one system must be selected to create a manifold."
            )

        if len(set(Manifolds)) != len(Manifolds):
            raise ValueError("Manifold names must be unique.")

        for name in Manifolds:
            if name not in self.System:
                raise KeyError(
                    f"Unknown quantum system '{name}'. "
                    f"Available systems: {list(self.System.keys())}"
                )

        self.Manifolds = Manifolds

        self.ManifoldDimensions = {
            name: int(self.System[name].Vdim)
            for name in self.Manifolds
        }

        self.ManifoldOffsets = {}
        self.ManifoldSlices = {}

        offset = 0

        for name in self.Manifolds:
            dim = self.ManifoldDimensions[name]
            self.ManifoldOffsets[name] = offset
            self.ManifoldSlices[name] = slice(offset, offset + dim)
            offset += dim

        # Direct sum => dimensions add.
        self.Vdim = int(offset)
        self.Ldim = int(self.Vdim ** 2)

        identity = np.eye(self.Vdim, dtype=np.complex64)
        self.id = QunObj(identity) if QuantumObject else identity

        self.ManifoldProjectors = {}

        for name in self.Manifolds:
            P = np.zeros(
                (self.Vdim, self.Vdim),
                dtype=np.complex64
            )

            sl = self.ManifoldSlices[name]
            dim = self.ManifoldDimensions[name]
            P[sl, sl] = np.eye(dim, dtype=np.complex64)

            Pout = QunObj(P) if QuantumObject else P
            self.ManifoldProjectors[name] = Pout
            setattr(self, f"{name}id", Pout)

        # Manifold bookkeeping is now complete.
        self.ManifoldCreated = True

        # Create:
        #
        #     QS.GS_Manifold.Az
        #     QS.ES_Manifold.Bz
        #     QS.SS_Manifold.id
        #
        # The original QS.GS.Az, QS.ES.Bz, ... are NOT modified.
        self.ManifoldViews = {}

        for name in self.Manifolds:

            system = self.System[name]
            view = _ManifoldView()

            view.Name = name
            view.LocalSystem = system
            view.LocalVdim = int(system.Vdim)
            view.Vdim = self.Vdim
            view.Ldim = self.Ldim
            view.Offset = self.ManifoldOffsets[name]
            view.Slice = self.ManifoldSlices[name]
            view.id = self.ManifoldProjectors[name]

            # QuantumSystem creates spin operators as attributes such as
            # Ax, Ay, Az, Ap, Am, Aid. Embed every square Hilbert-space
            # matrix of the local Vdim automatically.
            for attr_name, attr_value in vars(system).items():

                try:
                    A = self._Array(attr_value)
                except Exception:
                    continue

                if (
                    isinstance(A, np.ndarray)
                    and A.ndim == 2
                    and A.shape == (int(system.Vdim), int(system.Vdim))
                ):
                    setattr(
                        view,
                        attr_name,
                        self.EmbedOperator(
                            attr_value,
                            name,
                            QuantumObject=QuantumObject
                        )
                    )

            self.ManifoldViews[name] = view
            setattr(self, f"{name}_Manifold", view)

        return self

    def _CheckManifold(self, Manifold):
        if not self.ManifoldCreated:
            raise RuntimeError(
                "No manifold has been created. Call QS.CreateManifold() first."
            )

        if Manifold not in self.ManifoldSlices:
            raise KeyError(
                f"Unknown manifold '{Manifold}'. "
                f"Available manifolds: {self.Manifolds}"
            )

    @staticmethod
    def _Array(X):
        if isinstance(X, QunObj):
            return np.asarray(X.data)

        if hasattr(X, "data") and isinstance(X.data, np.ndarray):
            return np.asarray(X.data)

        return np.asarray(X)

    def EmbedOperator(self, Operator, Manifold, QuantumObject=True):
        """Embed a local manifold operator into the full direct-sum space."""

        self._CheckManifold(Manifold)
        O = self._Array(Operator)
        dim = self.ManifoldDimensions[Manifold]

        if O.shape != (dim, dim):
            raise ValueError(
                f"Operator for manifold '{Manifold}' must have shape "
                f"({dim}, {dim}); received {O.shape}."
            )

        O_full = np.zeros(
            (self.Vdim, self.Vdim),
            dtype=np.result_type(O.dtype, np.complex64)
        )

        sl = self.ManifoldSlices[Manifold]
        O_full[sl, sl] = O

        return QunObj(O_full) if QuantumObject else O_full

    def ManifoldState(self, Manifold, StateIndex, QuantumObject=True):
        """Create a basis ket in the complete manifold Hilbert space."""

        self._CheckManifold(Manifold)
        dim = self.ManifoldDimensions[Manifold]

        if not isinstance(StateIndex, (int, np.integer)):
            raise TypeError("StateIndex must be an integer.")

        if StateIndex < 0 or StateIndex >= dim:
            raise IndexError(
                f"State index {StateIndex} is outside manifold "
                f"'{Manifold}' with dimension {dim}."
            )

        ket = np.zeros((self.Vdim, 1), dtype=np.complex64)
        global_index = self.ManifoldOffsets[Manifold] + int(StateIndex)
        ket[global_index, 0] = 1.0

        return QunObj(ket) if QuantumObject else ket

    def ManifoldStateProjector(self, Manifold, StateIndex, QuantumObject=True):
        """Return |state><state| in the complete manifold Hilbert space."""

        ket = self.ManifoldState(
            Manifold,
            StateIndex,
            QuantumObject=False
        )

        P = ket @ ket.conj().T
        return QunObj(P) if QuantumObject else P

    def ManifoldQuantumNumbers(self, Manifold):
        """
        Return the allowed magnetic quantum numbers for every subsystem
        in a manifold.

        Examples
        --------
        For

            {"GS": {"A1": "NV_Zero", "A2": "NV_One"}}

        this returns conceptually

            {
                "A1": [0],
                "A2": [1, 0, -1]
            }

        The ordering follows the local QuantumSystem tensor-product basis.
        """

        self._CheckManifold(Manifold)

        system = self.System[Manifold]

        quantum_numbers = {}

        for spin in system.SpinDic:

            idx = system.SpinIndex[spin]
            s = float(system.slist[idx])

            # Same ordering used by QuantumSystem.SpinOperatorsSingleSpin():
            #
            #     m = s, s-1, ..., -s
            #
            m_values = np.arange(
                s,
                -s - 1.0,
                -1.0
            )

            quantum_numbers[spin] = [
                float(m)
                for m in m_values
            ]

        return quantum_numbers


    def ManifoldBasis(self, Manifold=None):
        """
        Print the basis states of one manifold or all manifolds.

        Parameters
        ----------
        Manifold : str, optional
            Name of a manifold such as "GS", "ES", or "SS".
            If None, basis states for all manifolds are printed.

        Examples
        --------
        QS.ManifoldBasis()
        QS.ManifoldBasis("ES")
        """

        if not self.ManifoldCreated:
            raise RuntimeError(
                "No manifold has been created. "
                "Call QS.CreateManifold() first."
            )

        if Manifold is None:
            manifolds_to_print = list(self.Manifolds)
        else:
            if Manifold not in self.Manifolds:
                raise KeyError(
                    f"Unknown manifold '{Manifold}'. "
                    f"Available manifolds: {self.Manifolds}"
                )
            manifolds_to_print = [Manifold]

        for manifold_name in manifolds_to_print:

            quantum_numbers = self.ManifoldQuantumNumbers(
                manifold_name
            )

            subsystem_names = list(
                quantum_numbers.keys()
            )

            basis_states = list(
                product(
                    *[
                        quantum_numbers[name]
                        for name in subsystem_names
                    ]
                )
            )

            dimension = self.ManifoldDimensions[
                manifold_name
            ]

            print("=" * 32)
            print(f"Manifold: {manifold_name}")
            print(f"Dimension: {dimension}")
            print("=" * 32)
            print()

            index_width = max(
                len("index"),
                len(str(max(dimension - 1, 0)))
            )

            column_widths = {}

            for name in subsystem_names:

                values = quantum_numbers[name]

                value_strings = [
                    self._FormatQuantumNumber(value)
                    for value in values
                ]

                column_widths[name] = max(
                    len(name),
                    max(
                        [len(value) for value in value_strings],
                        default=1
                    )
                )

            header = f"{'index':<{index_width}}"

            for name in subsystem_names:
                header += (
                    "    "
                    + f"{name:>{column_widths[name]}}"
                )

            print(header)
            print("-" * len(header))

            for index, state in enumerate(
                basis_states
            ):

                row = f"{index:<{index_width}}"

                for name, value in zip(
                    subsystem_names,
                    state
                ):
                    formatted = self._FormatQuantumNumber(
                        value
                    )

                    row += (
                        "    "
                        + f"{formatted:>{column_widths[name]}}"
                    )

                print(row)

            print()


    @staticmethod
    def _FormatQuantumNumber(Value):
        """
        Format magnetic quantum numbers for readable basis tables.

        Examples
        --------
        1.0   -> +1
        0.5   -> +0.5
        0.0   -> 0
        -0.5  -> -0.5
        -1.0  -> -1
        """

        value = float(Value)

        if np.isclose(
            value,
            0.0,
            atol=1.0e-12,
            rtol=0.0
        ):
            return "0"

        if np.isclose(
            value,
            round(value),
            atol=1.0e-12,
            rtol=0.0
        ):
            integer_value = int(
                round(value)
            )

            if integer_value > 0:
                return f"+{integer_value}"

            return str(integer_value)

        if value > 0:
            return f"+{value:g}"

        return f"{value:g}"


    def ManifoldBasisIndex(self, Manifold, State):
        """
        Convert physical subsystem quantum numbers to the zero-based
        local basis index of a manifold.

        Parameters
        ----------
        Manifold : str
            Name of the manifold, e.g. "GS", "ES", or "SS".

        State : dict
            Magnetic quantum number of every subsystem in that manifold.

            Example::

                {
                    "A1": 0,
                    "A2": 0
                }

        Returns
        -------
        int
            Local zero-based basis index.

        Notes
        -----
        PyOR constructs tensor-product operators with NumPy Kronecker
        products. Therefore the first subsystem is the slow index and
        the last subsystem is the fast index.

        For dimensions [d1, d2, ..., dn], the local index is

            (((i1*d2 + i2)*d3 + i3) ...).
        """

        self._CheckManifold(Manifold)

        if not isinstance(State, dict):
            raise TypeError(
                "State must be a dictionary of subsystem quantum numbers."
            )

        system = self.System[Manifold]
        expected_spins = list(system.SpinDic)

        missing = [
            spin
            for spin in expected_spins
            if spin not in State
        ]

        extra = [
            spin
            for spin in State
            if spin not in expected_spins
        ]

        if missing:
            raise ValueError(
                f"Missing subsystem quantum numbers for manifold "
                f"'{Manifold}': {missing}"
            )

        if extra:
            raise ValueError(
                f"Unknown subsystem names for manifold "
                f"'{Manifold}': {extra}"
            )

        quantum_numbers = self.ManifoldQuantumNumbers(
            Manifold
        )

        local_indices = []
        dimensions = []

        for spin in expected_spins:

            allowed = quantum_numbers[spin]
            value = float(State[spin])

            matches = [
                i
                for i, m in enumerate(allowed)
                if np.isclose(
                    value,
                    m,
                    atol=1.0e-8,
                    rtol=0.0
                )
            ]

            if len(matches) == 0:
                raise ValueError(
                    f"Invalid quantum number {State[spin]} "
                    f"for subsystem '{spin}' in manifold "
                    f"'{Manifold}'. Allowed values: {allowed}"
                )

            local_indices.append(matches[0])
            dimensions.append(len(allowed))

        # Flatten tensor-product indices using the same convention
        # as repeated np.kron in QuantumSystem.
        index = 0

        for local_index, dim in zip(
            local_indices,
            dimensions
        ):
            index = index * dim + local_index

        return int(index)


    def ManifoldStateQ(
        self,
        Manifold,
        State,
        QuantumObject=True
    ):
        """
        Create a full-manifold basis ket from physical subsystem
        quantum numbers.

        Example
        -------
        ket = QS.ManifoldStateQ(
            "ES",
            {
                "B1": 0.5,
                "B2": 0
            }
        )
        """

        index = self.ManifoldBasisIndex(
            Manifold,
            State
        )

        return self.ManifoldState(
            Manifold,
            index,
            QuantumObject=QuantumObject
        )


    def ManifoldStateProjectorQ(
        self,
        Manifold,
        State,
        QuantumObject=True
    ):
        """
        Return |state><state| using physical subsystem quantum numbers.
        """

        ket = self.ManifoldStateQ(
            Manifold,
            State,
            QuantumObject=False
        )

        P = ket @ ket.conj().T

        return QunObj(P) if QuantumObject else P


    def TransitionState(
        self,
        Initial,
        Final,
        QuantumObject=True
    ):
        """
        Construct the full-manifold transition operator

            |Final><Initial|

        using readable subsystem quantum numbers.

        Parameters
        ----------
        Initial : dict
            Must contain "Manifold" and every subsystem quantum number.

        Final : dict
            Must contain "Manifold" and every subsystem quantum number.

        Example
        -------
        T = QS.TransitionState(

            Initial={
                "Manifold": "GS",
                "A1": 0,
                "A2": 0
            },

            Final={
                "Manifold": "ES",
                "B1": 0.5,
                "B2": 0
            }

        )
        """

        if not isinstance(Initial, dict):
            raise TypeError(
                "Initial must be a dictionary."
            )

        if not isinstance(Final, dict):
            raise TypeError(
                "Final must be a dictionary."
            )

        if "Manifold" not in Initial:
            raise ValueError(
                "Initial must contain a 'Manifold' entry."
            )

        if "Manifold" not in Final:
            raise ValueError(
                "Final must contain a 'Manifold' entry."
            )

        initial_manifold = Initial["Manifold"]
        final_manifold = Final["Manifold"]

        initial_state = {
            key: value
            for key, value in Initial.items()
            if key != "Manifold"
        }

        final_state = {
            key: value
            for key, value in Final.items()
            if key != "Manifold"
        }

        initial_index = self.ManifoldBasisIndex(
            initial_manifold,
            initial_state
        )

        final_index = self.ManifoldBasisIndex(
            final_manifold,
            final_state
        )

        return self.Transition(
            initial_manifold,
            initial_index,
            final_manifold,
            final_index,
            QuantumObject=QuantumObject
        )


    def CollapseOperator(
        self,
        Initial,
        Final,
        Rate,
        QuantumObject=True
    ):
        """
        Construct a Lindblad collapse operator

            C = sqrt(Rate) |Final><Initial|.

        Parameters
        ----------
        Initial, Final : dict
            State definitions accepted by TransitionState().

        Rate : float
            Transition rate in s^-1.

        QuantumObject : bool, optional
            Return QunObj when True.

        Returns
        -------
        QunObj or ndarray
            Collapse operator in the complete manifold Hilbert space.
        """

        if not np.isscalar(Rate):
            raise TypeError(
                "Rate must be a scalar."
            )

        Rate = float(Rate)

        if Rate < 0:
            raise ValueError(
                "Transition rate must be non-negative."
            )

        T = self.TransitionState(
            Initial,
            Final,
            QuantumObject=False
        )

        C = np.sqrt(Rate) * T

        return QunObj(C) if QuantumObject else C


    def Transition(
        self,
        InitialManifold,
        InitialState,
        FinalManifold,
        FinalState,
        QuantumObject=True
    ):
        """
        Construct the inter-manifold transition operator |final><initial|.
        """

        ket_initial = self.ManifoldState(
            InitialManifold,
            InitialState,
            QuantumObject=False
        )

        ket_final = self.ManifoldState(
            FinalManifold,
            FinalState,
            QuantumObject=False
        )

        T = ket_final @ ket_initial.conj().T
        return QunObj(T) if QuantumObject else T
