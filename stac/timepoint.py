"""Provides class for creating and manipulating timepoints in circuits."""
from typing import Union, Iterator
from .operation import Operation


class Timepoint:
    """Class to create and manipulate timepoints."""

    def __init__(self,
                 new_op: Operation = None) -> None:
        self.operations: list[Operation] = []
        self.affected_qubits: set[tuple] = set()
        self.repeat_start = False
        self.repeat_end = False
        self.repeat_repetitions = None

        if new_op is not None:
            self.append(new_op)

    def __repr__(self) -> str:
        """Return a representation of the object."""
        return '\n'.join([str(op) for op in self.operations])

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def __iter__(self) -> Iterator[Operation]:
        """Return iterator of the Timepoint."""
        return self.operations.__iter__()

    def __getitem__(self, ind) -> Union[Operation, list[Operation]]:
        """Make Timepoint subscriptable."""
        return self.operations.__getitem__(ind)

    def __len__(self) -> int:
        """Return number of operations in the Timepoint."""
        return len(self.operations)

    def copy(self) -> 'Timepoint':
        """Return a copy of the Timepoint."""
        copied_tp = Timepoint()
        for op in self.operations:
            copied_tp.append(op.copy())

        return copied_tp

    def append(self,
               new_op: Operation) -> None:
        """
        Append operation to this Timepoint.

        Parameters
        ----------
        new_op : Operation
            Operation to append.

        Raises
        ------
        Exception
            If new_op can't be appended to current Timepoint.

        Returns
        -------
        None.

        """
        possible_intersections = self.affected_qubits \
            & new_op.affected_qubits
        if len(possible_intersections) == 0:
            self.affected_qubits |= new_op.affected_qubits

            self.operations.append(new_op.copy())
        else:
            raise Exception("Operations affects qubits already affected by\
                            this timepoint.")

    def can_append(self,
                   new_op: Operation) -> bool:
        """
        Check if an Operation can be appended to this Timepoint.

        Parameters
        ----------
        new_op : Operation
            Operation to be checked.

        Returns
        -------
        bool
            True if Operation can be appended, otherwise False.

        """
        possible_intersections = self.affected_qubits \
            & new_op.affected_qubits
        if len(possible_intersections) == 0:
            return True
        else:
            return False

    def rebase_qubits(self,
                      new_base: tuple) -> 'Timepoint':
        """
        Create Timepoint with new base address for all controls and targets.

        Parameters
        ----------
        new_base : tuple
            New base address. Must have length smaller than the shortest
            address within all controls and targets within qubits.

        Returns
        -------
        tp : Timepoint
            Timepoint with new base address.

        """
        tp = Timepoint()
        for op in self.operations:
            tp.append(op.rebase_qubits(new_base))
        return tp

    def can_add(self,
                other: 'Timepoint') -> bool:
        """
        Check if a Timepoint can be added to this Timepoint.

        Parameters
        ----------
        other : Timepoint
            The Timepoint to be checked.

        Returns
        -------
        bool
            True if other can be added, otherwise False.

        """
        for op in other:
            if not self.can_append(op):
                return False
        else:
            return True

    def __add__(self,
                other: 'Timepoint') -> 'Timepoint':
        """
        Create Timepoint that is sum of other Timepoint and this Timepoint.

        Parameters
        ----------
        other : Timepoint
            Timepoint to be added.

        Returns
        -------
        tp : Timepoint
            DESCRIPTION.

        """
        tp = self.copy()

        if self.can_add(other):
            for op in other:
                tp.append(op.copy())

        return tp

    def __iadd__(self,
                 other: 'Timepoint') -> 'Timepoint':
        """
        Add other Timepoint to this Timepoint.

        Parameters
        ----------
        other : Timepoint
            Timepoint to be added.

        Returns
        -------
        Timepoint
            Summed Timepoints.

        """
        if self.can_add(other):
            for op in other:
                self.append(op.copy())

        return self
