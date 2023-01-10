"""Provides class for creating and manipulating timepoints in circuits."""
from .operation import Operation


class Timepoint:
    """Class to create and manipulate timepoints."""

    def __init__(self, new_op=None):
        self.operations = []
        self.affected_qubits = set()
        if new_op is not None:
            self.append(new_op)

    def __repr__(self):
        return '\n'.join([str(op) for op in self.operations])

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        """Return iterator in the timepoint."""
        return self.operations.__iter__()

    def __getitem__(self, ind):
        """Make timepoint subscriptable."""
        return self.operations.__getitem__(ind)

    def __len__(self):
        """Return number of operations in the timepoint."""
        return len(self.operations)

    def copy(self):
        copied_tp = Timepoint()
        for op in self.operations:
            copied_tp.append(op.copy())
            
        return copied_tp

    def append(self, new_op):
        possible_intersections = self.affected_qubits \
            & new_op.affected_qubits
        if len(possible_intersections) == 0:
            self.affected_qubits |= new_op.affected_qubits

            self.operations.append(new_op.copy())
            return True
        else:
            return False

    def can_append(self, new_op):
        possible_intersections = self.affected_qubits \
            & new_op.affected_qubits
        if len(possible_intersections) == 0:
            return True
        else:
            return False

    def rebase_qubits(self, new_base):
        tp = Timepoint()
        for op in self.operations:
            tp.append(op.rebase_qubits(new_base))
        return tp

    def can_add(self, tp2):
        for op in tp2:
            if not self.can_append(op):
                return False
        else:
            return True

    def add(self, tp2):
        for op in tp2:
            if not self.can_append(op):
                return False
        else:
            for op in tp2:
                self.append(op)
            return True
