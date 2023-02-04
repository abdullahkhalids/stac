"""Provides classes to create and manipulate qubits."""
from typing import Union


class PhysicalQubit:
    """Class to create and manipulate physical qubits."""

    def __init__(self,
                 index: int,
                 coordinates: Union[int, tuple],
                 interactable_qubits: list[Union[int, tuple]]
                 ) -> None:
        """
        Construct a physical qubit.

        Parameters
        ----------
        index : int
            Index of qubits within its Register.
        coordinates : Union[int, tuple]
            The coordinate of the qubit.
        interactable_qubits : list[Union[int, tuple]]
            The qubits this qubit can interact with.

        """
        self.index = index
        self.coordinates = coordinates
        self.interactable_qubits = interactable_qubits


class VirtualQubit:
    """Class to create and manipulate virtual qubits."""

    def __init__(self,
                 level: int,
                 index_in_assigned_register: int,
                 assigned_register: tuple = None,
                 index_in_constituent_register: int = None,
                 constituent_register: tuple = None
                 ) -> None:
        """
        Construct a virtual qubit.

        Parameters
        ----------
        level : int
            The level of the Circuit this qubit is at.
        index_in_assigned_register : int
            The index within its assigned register.
        assigned_register : tuple, optional
            The address of the Register this qubit is part of. The default is
            None.
        index_in_constituent_register : int, optional
            The index within its constituent register. The default is None.
        constituent_register : tuple, optional
            Encoded qubits at level > 1 are made of a Register. This points to
            the address of that Register. The default is None.

        """
        self.level = level

        # The register this qubit is part of
        self.assigned_register = assigned_register
        self.index = index_in_assigned_register

        # The register this qubit is made of
        self.constituent_register = constituent_register
        self.index_in_constituent_register = index_in_constituent_register

        self.register_type = 'q'

    @property
    def index_in_assigned_register(self) -> int:
        """
        Get index in assigned register.

        Returns
        -------
        int
            Index in assigned register.

        """
        return self._index

    @index_in_assigned_register.setter
    def index_in_assigned_register(self,
                                   value: int) -> None:
        """
        Set index in assigned register.

        Parameters
        ----------
        value : int
            Value to set.

        """
        self._index = value

    @property
    def index(self):
        """
        Get index in assigned register.

        Returns
        -------
        int
            Index in assigned register.

        """
        return self._index

    @index.setter
    def index(self,
              value: int) -> None:
        """
        Set index in assigned register.

        Parameters
        ----------
        value : int
            Value to set.

        """
        self._index = value

    def copy(self) -> 'VirtualQubit':
        """
        Create copy of this register.

        Returns
        -------
        VirtualQubit
            The copy of self.

        """
        vq = VirtualQubit.__new__(VirtualQubit)
        vq.level = self.level

        vq.assigned_register = self.assigned_register
        vq.index = self.index_in_assigned_register

        vq.constituent_register = self.constituent_register
        vq.index_in_constituent_register = self.index_in_constituent_register

        vq.register_type = 'q'

        return vq
