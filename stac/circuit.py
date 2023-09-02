"""Provide a module to create and manipulate quantum circuits."""
from typing import Any, Iterator, Union, Optional, overload

from .operation import Operation
from .annotation import Annotation, AnnotationSlice
from .timepoint import Timepoint
from .qubit import PhysicalQubit  # , VirtualQubit
from .register import Register, QubitRegister, RegisterRegister
from .supportedoperations import _operations
from .instructionblock import InstructionBlock, AnnotationBlock

from IPython.display import display, SVG
from itertools import chain
import textwrap
import sys
import svg
import json
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
import stim
import copy
import bidict
import tabulate
tabulate.PRESERVE_WHITESPACE = True


def display_states(head,
                   *args: list[list]):
    """
    Display states as a pretty table.

    Parameters
    ----------
    head : List
        A list of headings for the table.
    *args : List[List]
        A list of states.

    Returns
    -------
    None.

    """
    if len(args) == 0:
        return
    else:
        comb_tab = copy.deepcopy(args[0])
        for i in range(len(args[0])):
            for v in args[1:]:
                comb_tab[i] += v[i][1:]

    # now delete unneeded lines
    smalltab = []
    for i in range(len(comb_tab)):
        t = comb_tab[i]
        if any(t[1:]):
            line = [t[0]]
            for val in t[1:]:
                real = val.real
                imag = val.imag
                if np.isclose(real, 0) and np.isclose(imag, 0):
                    line += [' ']
                elif np.isclose(imag, 0):
                    line += [f' {real:.3f}']
                elif np.isclose(real, 0):
                    line += [f'{imag:.3f}j']
                elif real > 0:
                    line += [f' {val:.3f}']
                else:
                    line += [f'{val:.3f}']
            smalltab.append(line)

    print(tabulate.tabulate(smalltab, headers=head, colalign=None))


class Circuit:
    """Class for creating and manipulating quantum circuits."""

    def __init__(self, *args: Any) -> None:
        """Construct a quantum circuit.

        Parameters
        ----------
        Register:
            If passed, then the Register is appended to the circuit.
        """
        self.instructions: InstructionBlock = InstructionBlock()
        self.annotations: AnnotationBlock = AnnotationBlock()
        self.annotations.append(AnnotationSlice())
        self.timepoints: list[Timepoint] = []
        self._cur_time = 0

        self.register = RegisterRegister('circuit', -2)
        self.register.index = 0

        self.register.append(RegisterRegister('level0', -1))
        self.register.structure = self._structure

        if (len(args) == 1
                and type(args[0]) in [RegisterRegister, QubitRegister]):
            self.append_register(args[0])

        self.base_address: Any = tuple()

        self.physical_register = Register()
        self.layout_map = None
        self.custom_gates = ''

    @staticmethod
    def simple(num_qubits: int) -> 'Circuit':
        """
        Create a simple circuit.

        In this circuit there is one register, and user can add operations by
        reference to an integer qubit index. For example, `append('H', 5)`.

        Parameters
        ----------
        num_qubits : int
            Number of qubits. The default is 0.

        Returns
        -------
        circ : Circuit
            An empty circuit.

        Examples
        --------
        >>> circ = stac.Circuit.simple(2)
        >>> circ.append('H', 0)
        >>> circ.append('CX', 0, 1)
        >>> circ
        0 H (0, 0, 0)
        1 CX (0, 0, 0) (0, 0, 1)
        """
        circ = Circuit()
        circ.append_register(QubitRegister('', 0, num_qubits))
        circ.base_address = (0, 0)
        return circ

    def __repr__(self) -> str:
        """Return a representation of the object."""
        label_len = len(str(len(self.instructions)-1))+1
        s = ''

        anns = self.annotations[0]
        if len(anns):
            st = textwrap.indent(str(anns), ' '*label_len) + '\n'
            s += st
        for i, tp in enumerate(self.instructions):
            st = textwrap.indent(str(tp), ' '*label_len)
            st = str(i).rjust(label_len-1) + st[label_len-1:] + '\n'
            s += st

            anns = self.annotations[i+1]
            if len(anns):
                st = textwrap.indent(str(anns), ' '*label_len) + '\n'
                s += st

        return s[:-1]

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def __len__(self) -> int:
        """Return number of operations in the quantum circuit."""
        return sum([len(ins) for ins in self.instructions]) \
            + sum([len(anns) for anns in self.annotations])

    def __iter__(self) -> Iterator:
        """Return iterator for the quantum circuit."""
        for ann in self.annotations[0]:
            yield ann
        for i in range(len(self.instructions)):
            for op in self.instructions[i]:
                yield op
            for ann in self.annotations[i+1]:
                yield ann

    def __getitem__(self,
                    ind: int
                    ) -> Operation:
        """
        Make circuit operations subscriptable.

        Parameters
        ----------
        ind : int
            Index of item to get.

        Raises
        ------
        IndexError
            If index out of bounds.

        Returns
        -------
        Operation
            The operation at index ind.

        """
        if ind >= 0:
            chunks = zip(self.instructions, self.annotations)
            range_parameter = ind + 1
        else:
            chunks = zip(
                map(reversed, reversed(self.annotations)),
                map(reversed, reversed(self.instructions)),
            )
            range_parameter = abs(ind)
        iterator = chain.from_iterable(chain.from_iterable(chunks))

        try:
            for _ in range(range_parameter):
                op = next(iterator)
        except StopIteration:
            raise IndexError("index out of range")

        return op

    def reverse(self) -> 'Circuit':
        """Return a circuit in which all operations are reversed."""
        rev_circuit = Circuit()
        rev_circuit.register = self.register.copy()
        # for tp in reversed(self.instructions):
        #     rev_circuit._append_tp(Timepoint())
        #     for op in reversed(tp.operations):
        #         rev_circuit.append(op, time=-1)
        rev_circuit.annotations.elements.clear()
        for i in range(len(self.instructions)-1, -1, -1):
            rev_circuit._append_tp(Timepoint())
            for ann in reversed(self.annotations[i+1].elements):
                rev_circuit.annotations[-1].append(ann)
            for op in reversed(self.instructions[i].operations):
                rev_circuit.instructions[-1].append(op)
        rev_circuit.annotations.append(AnnotationSlice())
        for ann in self.annotations[0].elements:
            rev_circuit.annotations[-1].append(ann)

        return rev_circuit

    def clear(self) -> None:
        """Remove all operations and annotations from circuit."""
        self.instructions: InstructionBlock = InstructionBlock()
        self.annotations: AnnotationBlock = AnnotationBlock()
        self.annotations.append(AnnotationSlice())
        self._cur_time = 0

    @property
    def cur_time(self) -> int:
        """
        Return time at which new operations will begin getting added.

        Returns
        -------
        int
            Current time.

        """
        return self._cur_time

    @cur_time.setter
    def cur_time(self,
                 new_time: int) -> None:
        """
        Set the current time in the circuit.

        Parameters
        ----------
        new_time : int
            The time to set.

        """
        if new_time >= 0:
            self._cur_time = new_time
        else:
            self._cur_time = len(self.instructions) + new_time

    def _standardize_addresses(self,
                               addresses: list[tuple]
                               ) -> list[tuple]:
        """
        Standardize input addresses with respect to base_address.

        All addresses must be targetted at the same level.

        Parameters
        ----------
        addresses : list[tuple]
            The addresses to standardize.

        Raises
        ------
        Exception
            If any address does not target a qubit.

        Returns
        -------
        tuple or list[tuple]
            Standardized address or list of standardized addresses.

        """
        standardized_addresses = []
        if self.base_address:
            level = self.base_address[0]
        else:
            level = addresses[0][0]
        for input_address in addresses:
            if type(input_address) is tuple:
                full_address = self.base_address + input_address
            elif type(input_address) is int:
                full_address = self.base_address + tuple([input_address])
            else:
                raise Exception('Not a valid address')

            if full_address[0] != level:
                raise Exception('Not all addresses are at the same level')

            self.register.check_address(full_address)
            standardized_addresses.append(full_address)

        if len(set(standardized_addresses)) != len(standardized_addresses):
            raise Exception('Some addresses were repeated')

        return standardized_addresses

    def _apply_encoded_operation(self,
                                 op: Operation,
                                 time: Optional[Union[int, list[int]]] = None
                                 ) -> None:
        """
        Apply encoded operation to circuit at level > 0.

        Parameters
        ----------
        op : Operation
            Operation to apply.
        time : int, optional
            Time at which to apply the Operation. The default is None, in which
            case the operation is appened via the default strategy.

        Raises
        ------
        Exception
            If the target register does not have a code attached.

        """
        t0 = self.register[op.targets[0]].constituent_register
        target_register = self.register[t0]
        circ = target_register.code.logical_circuits[op.name]
        if circ is None:
            raise Exception(f'No logical circuit for {op.name}')

        if op.num_affected_qubits == 1:
            for circ_op in circ:
                if circ_op.num_affected_qubits == 1:
                    self.append(circ_op.name,
                                t0+circ_op.targets[0][2:],
                                time=time)
                else:
                    self.append(circ_op.name,
                                t0+circ_op.targets[0][2:],
                                t0+circ_op.targets[1][2:],
                                time=time)
        else:
            t1 = self.register[op.targets[1]].constituent_register
            # what if circ_op is a one qubit operation
            for circ_op in circ:
                self.append(
                    op.name, t0+circ_op.targets[0][2:],
                    t1+circ_op.targets[1][2:], time=time)

    @overload
    def append(name: str,
               target: int | tuple,
               time: int | list[int] | None = None,
               ) -> None:
        ...

    @overload
    def append(name: str,
               control: int | tuple,
               target: int | tuple,
               time: int | list[int] | None = None,
               ) -> None:
        ...

    @overload
    def append(name: str,
               target: int | tuple,
               params: float | list[float] = None,
               time: int | list[int] | None = None,
               ) -> None:
        ...

    @overload
    def append(name: str,
               control: int | tuple,
               target: int | tuple,
               params: float | list[float] = None,
               time: int | list[int] | None = None,
               ) -> None:
        ...

    @overload
    def append(op: Operation
               ) -> None:
        ...

    @overload
    def append(ann: Annotation
               ) -> None:
        ...

    def append(self,
               *args,
               time=None,
               ):
        """
        Append a new operation to the circuit.

        Parameters
        ----------
        name : str
            Name of operation.
        targets : int or tuple
            The addresses of any target qubits.
        time : int or [1] or None, optional
            The time at which to append the operation. The default is None.
        params : float or list[float]
            If the gate is parameterized, this must be equal to the number of
            params passed.

        Raises
        ------
        Exception
            If Operation not valid, or cannot be appened.

        """
        # construct the operation if needed
        if len(args) == 1 and type(args[0]) is Operation:
            op = args[0]
            ins_type = 0
        elif len(args) == 1 and type(args[0]) is Annotation:
            ann = args[0]
            ins_type = 1
        else:
            if type(args[0]) is not str:
                raise Exception('Operation name must be str.')
            name = args[0].upper()

            # first do some type checking
            op_info = _operations.get(name, False)
            N = op_info["num_targets"]
            if not op_info:
                raise Exception('Not a known operation.')
            elif len(args) != N + 1 + op_info["is_parameterized"]:
                s = f'{name} takes {N} targets.'
                if op_info["is_parameterized"]:
                    s += f' And a {op_info["num_parameters"]} parameter list.'
                raise Exception(s)
            elif any(not isinstance(t, (int, tuple)) for t in args[1:N+1]):
                raise Exception('Target is not an int or tuple.')
            elif op_info['is_parameterized']:
                if (op_info['num_parameters'] == 1
                        and type(args[-1]) is not float):
                    raise Exception('parameter must be a float.')
                elif op_info['num_parameters'] > 1:
                    if (type(args[-1]) is not list
                            or len(args[-1]) != op_info['num_parameters']):
                        raise Exception(f'{name} needs \
{op_info["num_parameters"]} parameters')

            # now construct the operation
            if op_info['is_parameterized']:
                if op_info['num_parameters'] == 1:
                    parameters = [args[N+1]]
                else:
                    parameters = args[N+1]
            else:
                parameters = None

            if op_info['ins_type'] == 0:
                targets = self._standardize_addresses(list(args[1:N+1]))
                op = Operation(name, targets, parameters=parameters)
                ins_type = 0
            else:
                ann = Annotation(name)
                ins_type = 1

        # if needed add timepoints to circuit
        if time is None:
            while self._cur_time >= len(self.instructions):
                self._append_tp()

        # Insert annotation into the circuit
        if ins_type == 1:
            if len(self.instructions) == 0:
                self.annotations[0].append(ann)
            else:
                self.annotations[self.cur_time+1].append(ann)
                if ann.name == 'TICK':
                    self.cur_time += 1
            return

        # Insert instruction into the circuit
        if op.targets[0][0] == 0:
            if time is None:
                while self.cur_time < len(self.instructions):
                    if self.instructions[self.cur_time].can_append(op):
                        self.instructions[self.cur_time].append(op)
                        break
                    else:
                        self.cur_time += 1
                else:
                    tp = Timepoint(op)
                    self._append_tp(tp)
            elif time == [1]:
                tp = Timepoint(op)
                self._append_tp(tp)
            elif type(time) is int:
                while time >= len(self.instructions):
                    self._append_tp(Timepoint())
                if not self.instructions[time].can_append(op):
                    raise Exception('Cannot add operation to given timepoint.')
                else:
                    self.instructions[time].append(op)

        else:
            self._apply_encoded_operation(op, time=time)

    def geo_append(self,
                   *args,
                   time=None):
        """
        Append a new operation to the circuit, but using coordinates of qubits.

        Parameters
        ----------
        name : str
            Name of operation.
        targets : int or tuple
            The addresses of any target qubits.
        time : int or [1] or None, optional
            The time at which to append the operation. The default is None.
        params : float or list[float]
            If the gate is parameterized, this must be equal to the number of
            params passed.

        Raises
        ------
        Exception
            If Operation not valid, or cannot be appened.

        """
        if not self.layout_map:
            self.map_to_physical_layout()

        if type(args[0]) is not str:
            raise Exception('Operation name must be str.')
        name = args[0].upper()

        # first do some type checking
        op_info = _operations.get(name, False)
        N = op_info["num_targets"]
        if not op_info:
            raise Exception('Not a known operation.')
        elif len(args) != N + 1 + op_info["is_parameterized"]:
            s = f'{name} takes {N} targets.'
            if op_info["is_parameterized"]:
                s += f' And a {op_info["num_parameters"]} parameter list.'
            raise Exception(s)
        elif any(not isinstance(t, (int, tuple)) for t in args[1:N+1]):
            raise Exception('Target is not an int or tuple.')
        elif op_info['is_parameterized']:
            if (op_info['num_parameters'] == 1
                    and type(args[-1]) is not float):
                raise Exception('parameter must be a float.')
            elif op_info['num_parameters'] > 1:
                if (type(args[-1]) is not list
                        or len(args[-1]) != op_info['num_parameters']):
                    raise Exception(f'{name} needs \
{op_info["num_parameters"]} parameters')

        # now construct the operation
        if op_info['is_parameterized']:
            if op_info['num_parameters'] == 1:
                parameters = [args[N+1]]
            else:
                parameters = args[N+1]
        else:
            parameters = None

        if op_info['ins_type'] == 0:
            L = len(self.base_address)
            raw_targets = [self.layout_map.inverse[coord][L:]
                           for coord in args[1:N+1]]
            targets = self._standardize_addresses(raw_targets)
            op = Operation(name, targets, parameters=parameters)
            self.append(op)
        else:
            ann = Annotation(name)
            self.append(ann)

    def _append_tp(self,
                   tp: Timepoint = Timepoint(),
                   anns: AnnotationSlice = AnnotationSlice()) -> None:
        """
        Append Timepoint to circuit.

        Parameters
        ----------
        tp : Timepoint
            Timepoint to be appended. The default is an empty Timepoint.
        anns: AnnotationSlice
            AnnotationSlice to be appended. The default is an empty one.

        """
        self.instructions.append(tp.copy())
        self.annotations.append(anns.copy())

    def append_register(self,
                        register: Register
                        ) -> tuple:
        """
        Append a register to the circuit.

        Parameters
        ----------
        register : Register
            The register to be appended into the circuit. register.level should
            be set.

        Returns
        -------
        address: tuple
            Address of the appended register

        """
        level = register.level
        assert level is not None
        for i in range(len(self.register), level+1):
            self.register.append(RegisterRegister(f'level{i}', -1))

        register.index = len(self.register[level])
        self.register[level].append(register)

        return (register.level, register.index)

    def map_to_physical_layout(self,
                               layout: Optional[str] = 'linear'
                               ) -> list[list]:
        """
        Map the virtual qubits to physical qubits.

        Currently, there is only one inbuilt strategy, 'linear'. However,
        the user may write their own strategy for the mapping.

        Parameters
        ----------
        layout : str, optional
            Placeholder argument for now. The default is 'linear'.

        Returns
        -------
        layout_map: bidict.bidict
            bidict with items {virtual qubit address, physical qubit index}.
        """
        self.physical_register.elements = [PhysicalQubit(i, i, [])
                                           for i in range(self.num_qubits)]

        qa = self.register[0].qubit_addresses()
        self.layout_map = bidict.bidict()
        for i, address in enumerate(qa):
            self.register[0][address].constituent_register = \
                self.physical_register.elements[i]
            self.layout_map[(0,) + address] = i

        return self.layout_map

    def _structure(self,
                   depth: int = -1,
                   levels: Optional[Union[int, list]] = None
                   ) -> None:
        """
        Print register structure.

        Parameters
        ----------
        depth : int, optional
            The maximum depth to go to. The default is -1.
        levels : Optional[Union[int, list]], optional
            The levels of the circuit to display. The default is None.

        Raises
        ------
        TypeError
            If levels not correctly specfied.

        """
        if levels is None:
            for reg in self.register:
                reg.structure(depth)
        elif type(levels) is int and levels < len(self.register):
            self.register[levels].structure(depth)
        elif type(levels) is list:
            for i in levels:
                if i < len(self.registers):
                    self.register[i].structure(depth)
        else:
            raise TypeError('levels must be int or a list')

    @ property
    def num_qubits(self) -> int:
        """
        Determine number of qubits in circuit at level 0.

        Returns
        -------
        int
            Number of qubits.

        """
        return self.register[0].num_qubits

    def apply_circuit(self,
                      other: 'Circuit',
                      new_base: tuple,
                      time: Optional[Union[int, list[int]]] = None
                      ) -> None:
        """
        Apply other circuit to this circuit with a new base.

        Parameters
        ----------
        other : Circuit
            The circuit to be applied.
        new_base : tuple
            The base address at which to begin applying other circuit..
        time : int, optional
            Timepoint index at which to apply the other circuit. The default is
            None.

        Raises
        ------
        Exception
            Invalid time point.
        KeyError
            Cannot add circuits.

        """
        # qa = other.register.qubit_addresses()
        # L = len(new_base)
        # first_address_base = qa[0][0:L]
        # replace check with check on operations
        # if any(address[0:L] != first_address_base for address in qa):
        #     raise Exception('Base is not common to all qubits.')

        if time is None:
            for tp in other.timepoints:
                self._append_tp(tp.rebase_qubits(new_base))
        else:
            # decide where to start timeing the timepoints
            if time >= 0 and time < len(self.instructions):
                k = time
            elif time == 0 and len(self.instructions) == 0:
                k = time
            elif time < 0 and abs(time) <= len(self.instructions):
                k = len(self.instructions) + time
            else:
                raise KeyError('Invalid time point.')

            # check to make sure we can actually insert from this point
            for i, tp in enumerate(other.timepoints):
                if k+i < len(self.instructions):
                    if not self.instructions[k+i].can_add(
                            tp.rebase_qubits(new_base)):
                        raise Exception('Cannot add circuits.')

            # add the timepoints
            for i, tp in enumerate(other.timepoints):
                if k+i < len(self.instructions):
                    self.instructions[k+i] += tp.rebase_qubits(new_base)
                else:
                    self._append_tp(tp)

    # def start_repeat(self,
    #                  repetitions: int) -> None:
    #     """
    #     Start a repeat block with a new Timepoint.

    #     Parameters
    #     ----------
    #     repetitions : int
    #         The number of repetitions of the block.

    #     """
    #     self._append_tp(Timepoint())
    #     self.timepoints[-1].repeat_start = True
    #     self.timepoints[-1].repeat_repetitions = repetitions

    # def end_repeat(self) -> None:
    #     """Turn last Timepoint as end of repeat block."""
    #     self.timepoints[-1].repeat_end = True

    def __add__(self,
                other: 'Circuit'
                ) -> 'Circuit':
        """
        Add other circuit to this. Registers must match.

        Parameters
        ----------
        other : Circuit
            The circuit to be added to this one.

        Raises
        ------
        Exception:
            If registers are not compatible.

        Returns
        -------
        new_circuit : Circuit
            The composition of the two circuits.

        """
        if not self.register >= other.register:
            raise Exception("Registers not compatible.")

        new_circuit = copy.deepcopy(self)

        new_circuit.annotations[-1].elements += other.annotations[0].elements

        for i in range(len(other.instructions)):
            new_circuit._append_tp(other.instructions[i],
                                   other.annotations[i+1])
        new_circuit.cur_time = -1

        new_circuit.custom_gates = (self.custom_gates
                                    + '\n'
                                    + other.custom_gates)
        return new_circuit

    def __mul__(self,
                repetitions: int) -> 'Circuit':
        """
        Create a circuit which repeates repetitions times.

        Parameters
        ----------
        repetitions : int
            The number of repetitions.

        Returns
        -------
        new_circuit : Circuit
            The repeated circuit.

        """
        new_circuit = copy.deepcopy(self)

        new_circuit.timepoints[0].repeat_start = True
        new_circuit.timepoints[0].repeat_repetitions = repetitions

        new_circuit.timepoints[-1].repeat_end = True

        return new_circuit

    def qasm(self) -> str:
        """
        Convert circuit to qasm string.

        Returns
        -------
        qasm_str : str
            The qasm string of the circuit.

        """
        if not self.layout_map:
            self.map_to_physical_layout()
        qasm_str = ''

        for op in self:
            if _operations[op.name]['num_targets'] == 0:
                qasm_str += _operations[op.name]['qasm_str']
                continue

            t0 = self.register[op.targets[0]].constituent_register.index
            if op.num_affected_qubits == 1:
                if op.is_parameterized:
                    qasm_str += _operations[op.name]['qasm_str'].format(
                        p0=op.parameters[0], t0=t0)
                else:
                    qasm_str += _operations[op.name]['qasm_str'].format(t0=t0)
            else:
                t1 = self.register[op.targets[1]].\
                        constituent_register.index
                qasm_str += _operations[op.name]['qasm_str'].format(
                                                                t0=t0, t1=t1)

        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' \
            + self.custom_gates \
            + f'\nqreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\n\n' \
            + qasm_str

        return qasm_str

    def stim(self,
             clean: bool = False) -> str:
        """
        Convert circuit to a string that can be imported by stim.

        Parameters
        ----------
        clean : bool
            If True, then pass it through stim to compactify it.

        Returns
        -------
        stim_str : str
            A string suitable for importing by stim.

        """
        if not self.layout_map:
            self.map_to_physical_layout()
        stim_str = ''

        indent = ''
        for tp in self.instructions:
            for op in tp:
                opname = _operations[op.name]['stim_str']
                t0 = self.register[op.targets[0]].constituent_register.index
                if op.num_affected_qubits == 1:
                    stim_str += indent + f'{opname} {t0}\n'
                else:
                    t1 = self.register[op.targets[1]].\
                                            constituent_register.index
                    stim_str += indent + f'{opname} {t0} {t1}\n'

        if clean:
            stim_str = str(stim.Circuit(stim_str))
        return stim_str

    def quirk(self) -> None:
        """
        Convert circuit to a quirk circuit.

        Returns
        -------
        None.
        Prints a url that can opened in the browser.
        """
        validops = {'H', 'X', 'Y', 'Z', 'CX', 'CY', 'CZ'}
        cols = []

        if not self.layout_map:
            self.map_to_physical_layout()

        for op in self:
            if op.name in validops:
                L = [1 for i in range(self.num_qubits)]
                t0 = self.layout_map[op.targets[0]]
                if op.num_affected_qubits == 1:
                    L[t0] = op.name
                else:
                    t1 = self.layout_map[op.targets[1]]
                    L[t0] = "•"
                    L[t1] = op.name[1]

                cols.append(L)

        url = 'https://algassert.com/quirk#circuit={"cols":' + \
            json.dumps(cols, ensure_ascii=False) + '}'

        print(url)

    def simulate(self,
                 head: Optional[list[str]] = None,
                 incremental: bool = False,
                 return_state: bool = False,
                 print_state: bool = True
                 ) -> list[Any]:
        """
        Simulate the circuit using qiskit.

        Parameters
        ----------
        head : List, optional
            A list of strings that will act as headings. The default is None.
        incremental : bool, optional
            If true, circuit is simulated up to every TICK.
            The default is False.
        return_state : bool, optional
            If the state is returned by the fucntion. The default is False.
        print_state : bool, optional
            If the state is printed. The default is True.

        Returns
        -------
        tab : list
            The state.

        """
        n = self.num_qubits
        if head is None:
            head = ['basis', 'amplitude']

        tab = [[bin(i)[2:][-1::-1].ljust(n, '0')] for i in range(2**n)]

        cur_circ = copy.deepcopy(self)
        cur_circ.clear()

        for ind, op in enumerate(self):
            if ((op.name == 'TICK' and incremental)
                    or ind == len(self)-1):
                cur_circ.append(op)
                # cur_circ.append("I", n-1)
                qc = QuantumCircuit.from_qasm_str(cur_circ.qasm())
                job = execute(qc, Aer.get_backend('statevector_simulator'),
                              shots=1,
                              optimization_level=0)
                sv = job.result().get_statevector()
                amps = np.round(sv.data, 3)
                for i in range(2**n):
                    tab[i].append(amps[i])

            else:
                cur_circ.append(op)

        if print_state:
            display_states(head, tab)

        if return_state:
            return tab

    def sample(self,
               samples=1,
               return_sample: bool = False,
               print_sample: bool = True
               ) -> list[int]:
        """
        Return a sample from the circuit using stim.

        Parameters
        ----------
        return_sample : bool, optional
            If True, return the sample. The default is False.
        print_sample : bool, optional
            If True, print the sample. The default is True.

        Returns
        -------
        list
            The sample from the stim circuit.

        """
        stim_circ = stim.Circuit(self.stim())
        samples = stim_circ.compile_sampler().sample(samples)

        if print_sample:
            for s in samples:
                print(1*s, sep="")

        if return_sample:
            return samples

    def draw(self,
             medium: str = 'svg',
             filename: str = None,
             *,
             highlight_timepoints: Optional[bool] = False
             ) -> None:
        """
        Draw the circuit.

        Parameters
        ----------
        medium: str, optional
            Options are 'svg' or 'text'. Default is 'svg'.
        filename : str, optional
            If filename is provided, then the output will be written to the
            file. Otherwise, it will be displayed. The default is None.
        highlight_timepoints: bool, optional
            Only for medium='svg'. If True, each timepoint is highlighted.
            Default is False.

        """
        if medium == 'svg':
            self._draw_svg(filename,
                           highlight_timepoints)
        elif medium == 'text':
            self._draw_text(filename)

    def _draw_text(self,
                   filename: Optional[str] = None,
                   ) -> None:
        """
        Draw a text version of the circuit.

        Parameters
        ----------
        filename : str, optional
            If filename is provided, then the output will be written to the
            file. Otherwise, it will be printed out. The default is None.

        """
        dash = '─'
        space = ' '
        vert = '│'

        if not self.layout_map:
            self.map_to_physical_layout()

        num_qubits = self.num_qubits
        # lm = self.layout_map.copy()
        # lm.sort(key=lambda x: x[1])
        lm = self.layout_map
        address_label_len = max(map(len, map(lambda x: str(x), lm)))
        index_label_len = len(str(num_qubits))
        label_len = address_label_len + index_label_len+3
        circ_disp = [[str(address).ljust(address_label_len)
                     + ' : '
                     + str(index).rjust(index_label_len)
                     + space]
                     for address, index in lm.items()]
        circ_disp2 = [list(space*(label_len+1))
                      for _ in range(num_qubits)]

        # circ_tp_line = [space*(label_len+1)]

        for k, tp in enumerate(self.instructions):
            tp = self.instructions[k]
            slices = [[]]
            slices_touched_qubits = [set()]
            for op in tp.operations:

                t0 = self.register[op.targets[0]].constituent_register.index

                if op.num_affected_qubits == 1:
                    touched_by_op = set([t0])
                else:
                    t1 = self.register[op.targets[1]
                                       ].constituent_register.index
                    touched_by_op = set(list(range(t1, t0))
                                        + list(range(t0, t1)))

                for s in range(len(slices)):
                    if touched_by_op.isdisjoint(slices_touched_qubits[s]):
                        slices[s].append(op)
                        slices_touched_qubits[s].update(touched_by_op)
                        break
                else:
                    slices.append([op])
                    slices_touched_qubits.append(touched_by_op)

            # circ_tp_line.append('⍿' + space*(3*(len(slices)-1)+2))

            for sl in slices:
                touched_places = []

                for op in sl:
                    t0 = self.register[op.targets[0]].\
                        constituent_register.index
                    draw_text = _operations[op.name]['draw_text']

                    if op.num_affected_qubits == 1:
                        s = dash + draw_text[0] + dash
                        circ_disp[t0].append(s)
                        circ_disp2[t0].append(space*3)
                        touched_places.append(t0)

                    elif op.num_affected_qubits == 2:
                        t1 = self.register[op.targets[1]
                                           ].constituent_register.index
                        vert_places = list(range(t1, t0)) + list(range(t0, t1))
                        for i in range(num_qubits):
                            if i == t0:
                                circ_disp[i].append(
                                    dash + draw_text[0] + dash)
                                if i == vert_places[0]:
                                    circ_disp2[i].append(space + vert + space)
                                else:
                                    circ_disp2[i].append(space*3)
                                touched_places.append(i)
                            elif i == t1:
                                circ_disp[i].append(
                                    dash + draw_text[1] + dash)
                                if i == vert_places[0]:
                                    circ_disp2[i].append(space + vert + space)
                                else:
                                    circ_disp2[i].append(space*3)
                                touched_places.append(i)
                            elif i in vert_places[1:]:
                                circ_disp[i].append(dash + '┼' + dash)
                                circ_disp2[i].append(space + vert + space)
                                touched_places.append(i)

                for i in range(num_qubits):
                    if i not in set(touched_places):
                        circ_disp[i].append(dash*3)
                        circ_disp2[i].append(space*3)

            for ann in self.annotations[k+1]:
                if ann.name == 'TICK':
                    for i in range(num_qubits-1):
                        circ_disp[i].append('⸽')
                        circ_disp2[i].append('⸽')
                    circ_disp[num_qubits-1].append('⸽')

        if filename is None:
            file = sys.stdout
        else:
            file = open(filename, 'w')

        # print(''.join(circ_tp_line), file=file, flush=True)
        for line1, line2 in zip(circ_disp, circ_disp2):
            print(''.join(line1), file=file)
            print(''.join(line2), file=file, flush=True)

    def _draw_svg(self,
                  filename: Optional[str] = None,
                  highlight_timepoints=False
                  ) -> None:
        """
        Draw a svg version of the circuit.

        Parameters
        ----------
        filename : Optional[str], optional
            If filename is provided, then the output will be written to the
            file. Otherwise, it will be printed out. The default is None.
        highlight_timepoints : TYPE, optional
            If True, each timepoint is highlighted. The default is False.

        """
        if not self.layout_map:
            self.map_to_physical_layout()

        y_shift = 20
        bxs = 15
        bh = 26
        bys = -bh/2
        ly1s = 3
        ly2s = -bh/2

        el = []
        wirey = [y_shift + i*46 for i in range(self.num_qubits)]

        num_qubits = self.num_qubits
        lm = self.layout_map
        address_label_len = max(map(len, map(lambda x: str(x), lm)))
        index_label_len = len(str(num_qubits))
        nbspace = '&#160;'
        labels = [str(address)
                  + nbspace*(address_label_len-len(str(address))+1)
                  + ':'
                  + nbspace*(index_label_len-len(str(index))+1)
                  + str(index)
                  for address, index in lm.items()]
        x0 = (address_label_len+index_label_len+3)*8

        # add labels
        for i in range(num_qubits):
            el.append(svg.Text(x=0, y=wirey[i],
                               text=labels[i],
                               class_=["labeltext"],
                               dominant_baseline='central'
                               ))

        time = 0
        max_width = 0
        slice_x = x0
        recs = []
        highlight_class = "tp_highlight1"

        for k, tp in enumerate(self.instructions):
            highlight_class = "tp_highlight1" \
                if highlight_class == "tp_highlight2" else "tp_highlight2"
            start_time = slice_x

            slices = [[]]
            slices_touched_qubits = [set()]
            for op in tp.operations:

                t0 = self.register[op.targets[0]].constituent_register.index

                if op.num_affected_qubits == 1:
                    touched_by_op = set([t0])
                elif op.num_affected_qubits == 2:
                    t1 = self.register[op.targets[1]
                                       ].constituent_register.index
                    touched_by_op = set(list(range(t1, t0))
                                        + list(range(t0, t1)))

                for s in range(len(slices)):
                    if touched_by_op.isdisjoint(slices_touched_qubits[s]):
                        slices[s].append(op)
                        slices_touched_qubits[s].update(touched_by_op)
                        break
                else:
                    slices.append([op])
                    slices_touched_qubits.append(touched_by_op)

            for sl in slices:

                for op in sl:
                    t0 = self.register[op.targets[0]].\
                        constituent_register.index
                    draw_img = _operations[op.name]['draw_img']

                    if op.num_affected_qubits == 1:
                        width = len(draw_img[0])*18+10
                        el += [
                            svg.Rect(
                                x=slice_x+bxs, y=wirey[t0]+bys,
                                width=width, height=bh,
                                class_=["gaterect"],
                            ),
                            svg.Text(
                                x=slice_x+bxs+width/2, y=wirey[t0]+bys+bh/2,
                                text=draw_img[0],
                                class_=["gatetext"],
                                text_anchor='middle',
                                dominant_baseline='central'
                            )]

                    elif op.num_affected_qubits == 2:
                        t1 = self.register[op.targets[1]
                                           ].constituent_register.index

                        name_label = draw_img[1]
                        width = len(name_label)*16 + 12

                        el += [
                            svg.Circle(
                                cx=slice_x+bxs+width/2, cy=wirey[t0], r=3,
                                class_=["control1"]
                            ),
                            svg.Line(
                                x1=slice_x+bxs+width/2, x2=slice_x+bxs+width/2,
                                y1=wirey[t0]+ly1s, y2=wirey[t1]+ly2s,
                                class_=["controlline"]
                            ),
                            svg.Rect(
                                x=slice_x+bxs, y=wirey[t1]+bys,
                                width=width, height=bh,
                                class_=["gaterect"]
                            ),
                            svg.Text(
                                x=slice_x+bxs+width/2, y=wirey[t1]+bys+bh/2,
                                text=name_label,
                                class_=["gatetext"],
                                text_anchor='middle',
                                dominant_baseline='central'
                            )
                        ]
                    max_width = max(max_width, width)
                time += 1
                slice_x += max_width+bxs

            for ann in self.annotations[k+1]:
                if ann.name == 'TICK':
                    el += [
                        svg.Line(
                            x1=slice_x+0.5*bxs, x2=slice_x+0.5*bxs,
                            y1=0, y2=wirey[-1]+y_shift,
                            class_=["tickline"]
                            )]

            if highlight_timepoints:
                recs.append(svg.Rect(
                    x=start_time+bxs/2, y=0,
                    width=slice_x-start_time, height=wirey[-1]+y_shift,
                    class_=[highlight_class]))

        fig_width = slice_x+bxs
        fig_height = wirey[-1]+bh

        el = [svg.Style(
            text="""
            .backgroundrect {fill: white;}
            .labeltext { font-family: Bitstream Vera Sans Mono;
                        font-size: 12px; font-weight: 400; fill: black;}
            .qubitline { stroke: black; stroke-width: 2; }
            .gatetext { font-family: Latin Modern Math, Cambria Math;
                        font-size: 20px; font-weight: 400; fill: black;}
            .gaterect { fill: white; stroke: black; stroke-width: 2 }
            .control1 { fill: black; stroke: black; stroke-width: 1 }
            .controlline { stroke: black; stroke-width: 2}
            .tickline { stroke: black; stroke-width: 0.75; stroke-dasharray: 6,3}
            .tp_highlight1 { fill: red; opacity: 0.2;}
            .tp_highlight2 { fill: blue; opacity: 0.2;}""")] + \
            [svg.Rect(x=0, y=0,
                      width=fig_width, height=fig_height,
                      class_=["backgroundrect"])] + \
            recs + \
            [svg.Line(
                x1=x0+0, x2=slice_x+bxs,
                y1=y, y2=y,
                class_=["qubitline"]) for y in wirey] + el

        s = svg.SVG(
            width=fig_width,
            viewBox=f"0 0 {fig_width} {fig_height}",
            elements=el,
        )

        if filename is None:
            display(SVG(s.as_str()))
        else:
            with open(filename, 'w') as f:
                f.write(s.as_str())
