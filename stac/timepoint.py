"""Provides class for creating and manipulating timepoints in circuits."""
from .operation import Operation


class Timepoint:
    """Class to create and manipulate timepoints."""

    def __init__(self, new_op=None):
        self.operations = []
        self.affected_qubits = set()
        if new_op is not None:
            self.append(new_op)

    def append(self, new_op):
        possible_intersections = self.affected_qubits \
            & new_op.affected_qubits
        if len(possible_intersections) == 0:
            self.affected_qubits |= new_op.affected_qubits

            self.operations.append(new_op)
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

    def __str__(self):
        return '\n'.join([str(op) for op in self.operations])

    def __iter__(self):
        """Return iterator for the quantum circuit."""
        return self.operations.__iter__()

    def remap_qubits(self, qubit_map):
        tp = Timepoint()
        for op in self.operations:
            tp.append(op.remap_qubits(qubit_map))
        return tp

    def add(self, tp2):
        for op in tp2:
            if not self.can_append(op):
                return False
        else:
            for op in tp2:
                self.append(op)
            return True
