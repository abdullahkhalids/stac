"""Provide a class for operations in circuits."""
from typing import Optional


class Operation:
    """Class to represent operations in circuits."""

    _draw_str_target_dic: dict = {'CX': '⊕',
                                  'CY': 'Y',
                                  'CZ': '●',
                                  'MR': 'm'}
    
    _draw_img_target_dic: dict = {'CX': 'X',
                                  'CY': 'Y',
                                  'CZ': 'Z',}

    def __init__(self,
                 name: str,
                 targets: list[tuple],
                 controls: Optional[list[tuple]] = None,
                 classical_control: None = None) -> None:
        """
        Construct Operation object.

        Parameters
        ----------
        name : str
            Name of operation.
        targets : list[tuple]
            List of addresses that the operation targets.
        controls : Optional[list[tuple]], optional
            If this is a quantum-controlled operation, then this is a list of
            addresses that control the operation. The default is None.
        classical_control : None, optional
            This parameter is unused at the moment. The default is None.

        This contructor does no checks on whether the name, controls or targets
        are valid. These checks should be done when appending the operation
        to the circuit.
        """
        self.name = name
        self.targets = targets.copy()
        self.num_affected_qubits = len(targets)
        self.affected_qubits = set(targets.copy())

        self.draw_str_target = self._draw_str_target_dic.get(name, name[0])

        if controls is not None:
            self.is_controlled = True
            self.controls = controls
            self.control_state = '1'*len(controls)
            self.num_affected_qubits += len(controls)
            self.affected_qubits |= set(controls.copy())
            self.draw_str_control = '●'
            self.draw_str_target = self._draw_str_target_dic.get(name, name[1])
        else:
            self.is_controlled = False

    def __repr__(self) -> str:
        """Return a representation of the object."""
        if self.is_controlled:
            return ' '.join([self.name,
                            str(self.controls[0]),
                            str(self.targets[0])])
        else:
            return ' '.join([self.name,
                            str(self.targets[0])])

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def copy(self) -> 'Operation':
        """Return copy of class."""
        copied_op = Operation(self.name,
                              [],
                              controls=None,
                              classical_control=None)

        copied_op.targets = self.targets
        copied_op.num_affected_qubits = self.num_affected_qubits
        copied_op.affected_qubits = self.affected_qubits

        copied_op.draw_str_target = self.draw_str_target

        if self.is_controlled:
            copied_op.is_controlled = self.is_controlled
            copied_op.controls = self.controls
            copied_op.control_state = self.control_state
            copied_op.draw_str_control = self.draw_str_control
        else:
            copied_op.is_controlled = self.is_controlled

        return copied_op

    def rebase_qubits(self,
                      new_base: tuple) -> 'Operation':
        """
        Create Operation with new base address of the controls and targets.

        Parameters
        ----------
        new_base : tuple
            The base address to replace the existing base. This can be any
            length shorter than the length of the smallest address within the
            controls and targets.

        Returns
        -------
        Operation
            A new Operation with new base address.

        """
        L = len(new_base)
        new_targets = [new_base + q[L:] for q in self.targets]
        if self.is_controlled:
            new_controls = [new_base + q[L:] for q in self.controls]
        else:
            new_controls = None
        return Operation(self.name, new_targets, controls=new_controls)
