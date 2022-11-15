"""Provides a set of classes to define registers of qubits."""
from .qubit import CircuitQubit


class Register:
    """Class to create and manipulate registers."""

    def __init__(self):

        # self.index = None
        # functional type
        self.register_type = None

        self.elements = []

        self.level = None


class QubitRegister(Register):
    """Class to manipulate registers made out of circuit qubits."""

    def __init__(self,
                 register_type,
                 level,
                 num_qubits,
                 index=None):

        self.register_type = register_type
        self.level = level
        qubit_list = []
        for i in range(num_qubits):
            q = CircuitQubit(self.level,
                             i)
            qubit_list.append(q)

        self.elements = tuple(qubit_list)

        self.index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        for qubit in self.elements:
            qubit.assigned_register = self._index


class RegisterRegister(Register):
    """Class to manipulate registers made out of subregisters."""

    def __init__(self,
                 register_type,
                 level,
                 *args):

        self.register_type = register_type
        self.level = level

        if len(args) == 1 and type(args[0]) is list:
            self.elements = tuple(args[0])
        else:
            self.elements = args

        for i, reg in enumerate(self.elements):
            reg.index = i

        self.index = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
