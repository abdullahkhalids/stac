"""Provides classes to create and manipulate qubits."""


class PhysicalQubit:
    """Class to create and manipulate physical qubits."""

    def __init__(self, index, coordinates, interactable_qubits):
        self.index = index
        self.coordinates = coordinates
        self.interactable_qubits = interactable_qubits


class VirtualQubit:
    """Class to create and manipulate virtual qubits."""

    def __init__(self,
                 level,
                 index_in_assigned_register,
                 assigned_register=None,
                 index_in_constituent_register=None,
                 constituent_register=None):

        self.level = level

        # The register this qubit is part of
        self.assigned_register = assigned_register
        self.index = index_in_assigned_register

        # The register this qubit is made of
        self.constituent_register = constituent_register
        self.index_in_constituent_register = index_in_constituent_register

        self.register_type='q'

    @property
    def index_in_assigned_register(self):
        return self._index

    @index_in_assigned_register.setter
    def index_in_assigned_register(self, value):
        self._index = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
