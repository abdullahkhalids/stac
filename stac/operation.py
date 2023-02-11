"""Provide a class for operations in circuits."""
from typing import Optional


class Operation:
    """Class to represent operations in circuits."""

    def __init__(self,
                 name: str,
                 targets: list[tuple],
                 parameters: list[float] = None
                 ) -> None:
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
        are valid. These checks should be done before appending the operation
        to the circuit.
        """
        # Todo Add classical controls
        # Any changes here should be reflected in copy()
        self.name = name
        self.targets = targets.copy()
        self.num_affected_qubits = len(targets)
        self.affected_qubits = set(targets.copy())

        # self.draw_str_target = self._draw_str_target_dic.get(name, name[0])

        # if controls is not None:
            # self.is_controlled = True
            # self.controls = controls
            # self.control_state = '1'*len(controls)
            # self.num_affected_qubits += len(controls)
            # self.affected_qubits |= set(controls.copy())
            # self.draw_str_control = '●'
            # self.draw_str_target = self._draw_str_target_dic.get(name, name[1])
        # else:
            # self.is_controlled = False

        if parameters is not None:
            self.is_parameterized = True
            self.parameters = parameters
            self.num_parameters = len(parameters)
        else:
            self.is_parameterized = False

    def __repr__(self) -> str:
        """Return a representation of the object."""
        s = self.name
        if self.is_parameterized:
            s += '(' + str(self.parameters)[1:-1] + ')'
        # if self.is_controlled:
        #     s += ' ' + str(self.controls[0])
        s += ' ' + ' '.join([str(t) for t in self.targets])
        return s

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def copy(self) -> 'Operation':
        """Return copy of class."""
        copied_op = Operation.__new__(Operation)
        
        copied_op.name = self.name
        copied_op.targets = self.targets
        copied_op.num_affected_qubits = self.num_affected_qubits
        copied_op.affected_qubits = self.affected_qubits

        # copied_op.draw_str_target = self.draw_str_target

        # copied_op.is_parameterized = self.is_parameterized
        # if self.is_controlled:
        #     copied_op.controls = self.controls
        #     copied_op.control_state = self.control_state
        #     copied_op.draw_str_control = self.draw_str_control

        copied_op.is_parameterized = self.is_parameterized
        if self.is_parameterized:
            copied_op.parameters = self.parameters
            copied_op.num_parameters = self.num_parameters
            

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
