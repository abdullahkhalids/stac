"""Provide a module to create and manipulate quantum circuits."""
from typing import Any, Iterator, Union, Optional, Callable
from .operation import Operation
from .timepoint import Timepoint
from .qubit import PhysicalQubit  # , VirtualQubit
from .register import Register, QubitRegister, RegisterRegister
from .supportedoperations import _zero_qubit_operations,\
    _one_qubit_operations, _two_qubit_operations

import textwrap
import sys

import svg
import json
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
import stim
import copy
import tabulate
tabulate.PRESERVE_WHITESPACE = True
from IPython.display import display, SVG


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
        self.timepoints: list[Timepoint] = []
        self._cur_time = 0

        self.register = RegisterRegister('circuit', -2)
        self.register.index = 0

        self.register.append(RegisterRegister('level0', -1))
        self.register.structure = self._structure  # type: ignore

        if (len(args) == 1
                and type(args[0]) in [RegisterRegister, QubitRegister]):
            self.append_register(args[0])

        self.base_address: Any = tuple()

        self._layout_map = None
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
        """
        circ = Circuit()
        circ.append_register(QubitRegister('', 0, num_qubits))
        circ.base_address = (0, 0)
        return circ

    def __repr__(self) -> str:
        """Return a representation of the object."""
        label_len = len(str(len(self.timepoints)-1))+1
        s = ''
        for i, tp in enumerate(self.timepoints):
            st = textwrap.indent(str(tp), ' '*label_len)
            st = str(i).rjust(label_len-1) + st[label_len-1:] + '\n'
            s += st

        return s

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def __iter__(self) -> Iterator:
        """Return iterator for the quantum circuit."""
        for tp in self.timepoints:
            for op in tp:
                yield op

    def __getitem__(self,
                    ind: int) -> Operation:
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
            iterator = self.timepoints.__iter__()
            compare = int.__gt__
            inc = int.__add__
        else:
            iterator = reversed(self.timepoints)
            compare = int.__le__
            inc = int.__sub__

        s = 0
        for tp in iterator:
            L = len(tp)
            if compare(inc(s, L), ind):
                return tp[ind-s]  # type: ignore
            else:
                s = inc(s, L)
        else:
            raise IndexError('circuit index out of range')

    def __len__(self) -> int:
        """Return number of operations in the quantum circuit."""
        return sum([len(tp) for tp in self.timepoints])

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
        self._cur_time = new_time
        while self._cur_time >= len(self.timepoints):
            self._append_tp(Timepoint())

    def _standardize_addresses(self,
                               addresses: Union[tuple, list[tuple]]
                               ) -> Union[tuple, list[tuple]]:
        """
        Standardize input addresses with respect to base_address.

        All addresses must be targetted at the same level.

        Parameters
        ----------
        addresses : tuple or list[tuple]
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
        if type(addresses) is not list:
            not_list = True
            addresses = [addresses]  # type: ignore
        else:
            not_list = False

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

        if not_list:
            return standardized_addresses[0]
        else:
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
        crt = self.register[op.targets[0]].constituent_register  # type: ignore
        target_register = self.register[crt]  # type: ignore
        circ = target_register.code.logical_circuits[op.name]  # type: ignore
        if circ is None:
            raise Exception(f'No logical circuit for {op.name}')

        if op.is_controlled:
            crc = self.register[
                op.controls[0]].constituent_register  # type: ignore

        if not op.is_controlled:
            for circ_op in circ:
                if not circ_op.is_controlled:
                    self.append(circ_op.name,
                                crt+circ_op.targets[0][2:],
                                time=time)
                else:
                    self.append(circ_op.name,
                                crt+circ_op.controls[0][2:],
                                crt+circ_op.targets[0][2:],
                                time=time)
        else:
            for circ_op in circ:
                self.append(
                    op.name, crc+circ_op.controls[0][2:],
                    crt+circ_op.targets[0][2:], time=time)

    def append(self,
               *args: Any,
               time: Optional[Union[int, list[int]]] = None) -> None:
        """
        Append a new operation to the circuit.

        Parameters
        ----------
        name : str
            Name of operation.
        controls and target : int or tuple
            The address of any control or target qubits.
        time : int or [1], optional
            The time at which to append the operation. The default is None.

        Raises
        ------
        Exception
            If Operation not valid, or cannot be appened.

        """
        # construct the operation if needed
        if len(args) == 1 and type(args[0]) is Operation:
            op = args[0]
        elif type(args[0]) is str:
            name = args[0].upper()
            if len(args) == 1 and name in _zero_qubit_operations:
                pass
            elif len(args) == 2 and name in _one_qubit_operations:
                target = self._standardize_addresses(args[1])
                op = Operation(name, [target])  # type: ignore
            elif len(args) == 3 and name in _two_qubit_operations:
                [control, target] = self._standardize_addresses([args[1],
                                                                 args[2]])
                op = Operation(name, [target], [control])
            else:
                raise Exception('Not a valid operation')
        else:
            raise Exception('Not a valid operation')

        # Insert the operation into the circuit
        if op.targets[0][0] == 0:
            if time is None:
                while self.cur_time < len(self.timepoints):
                    if self.timepoints[self.cur_time].can_append(op):
                        self.timepoints[self.cur_time].append(op)
                        break
                    else:
                        self.cur_time += 1
                else:
                    tp = Timepoint(op)
                    self.timepoints.append(tp)

                # if len(self.timepoints) == 0 \
                #         or not self.timepoints[-1].can_append(op):
                #     tp = Timepoint(op)
                #     self.timepoints.append(tp)
                # else:
                #     self.timepoints[-1].append(op)
            elif time == [1]:
                tp = Timepoint(op)
                self.timepoints.append(tp)
            elif type(time) is int:
                while time >= len(self.timepoints):
                    self._append_tp(Timepoint())
                if not self.timepoints[time].can_append(op):
                    raise Exception('Cannot add operation to given timepoint.')
                else:
                    self.timepoints[time].append(op)

        else:
            self._apply_encoded_operation(op, time=time)

    def _append_tp(self,
                   tp: Timepoint) -> None:
        """
        Append Timepoint to circuit.

        Parameters
        ----------
        tp : Timepoint
            Timepoint to be appended.

        """
        self.timepoints.append(tp.copy())

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
        layout_map: list[list]
            List of the pairs [virtual qubit address, physical qubit index].
        """
        self.physical_register = Register()

        # x = list(range(self.num_qubits))
        self.physical_register.elements = [PhysicalQubit(i, i, [])
                                           for i in range(self.num_qubits)]

        qa = self.register[0].qubit_addresses()
        self._layout_map = []
        for i, address in enumerate(qa):
            self.register[0][address].constituent_register = \
                self.physical_register.elements[i]
            self._layout_map.append([(0,) + address, i])

        return self._layout_map

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
            if time >= 0 and time < len(self.timepoints):
                k = time
            elif time == 0 and len(self.timepoints) == 0:
                k = time
            elif time < 0 and abs(time) <= len(self.timepoints):
                k = len(self.timepoints) + time
            else:
                raise KeyError('Invalid time point.')

            # check to make sure we can actually insert from this point
            for i, tp in enumerate(other.timepoints):
                if k+i < len(self.timepoints):
                    if not self.timepoints[k+i].can_add(
                            tp.rebase_qubits(new_base)):
                        raise Exception('Cannot add circuits.')

            # add the timepoints
            for i, tp in enumerate(other.timepoints):
                if k+i < len(self.timepoints):
                    self.timepoints[k+i] += tp.rebase_qubits(new_base)
                else:
                    self._append_tp(tp)

    def start_repeat(self,
                     repetitions: int) -> None:
        """
        Start a repeat block with a new Timepoint.

        Parameters
        ----------
        repetitions : int
            The number of repetitions of the block.

        """
        self._append_tp(Timepoint())
        self.timepoints[-1].repeat_start = True
        self.timepoints[-1].repeat_repetitions = repetitions

    def end_repeat(self) -> None:
        """Turn last Timepoint as end of repeat block."""
        self.timepoints[-1].repeat_end = True

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

        new_circuit._append_tp(Timepoint())

        for tp in other.timepoints:
            for op in tp:
                new_circuit.append(op)

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
        qasm_str = ''

        for op in self:
            t = self.register[op.targets[0]].constituent_register.index
            if op.name[:7] == 'X_ERROR':
                continue
            elif op.name == 'R':
                op_str = f'reset q[{t}];\n'
            elif op.name == 'MR' or op.name == 'M':
                op_str = f'measure q[{t}] -> c[{t}];\n'
            elif op.name == 'I':
                op_str = f'id q[{t}];\n'
            # elif op.name == 'TICK':
            #     op_str = 'barrier '
            #     for i in range(t, op[3]+1):
            #         op_str += f'q[{i}],'
            #     op_str = op_str[:-1] + ';\n'
            else:
                op_str = op.name.lower() + ' '
                # followed by one or two arguments
                if not op.is_controlled:
                    op_str += f'q[{t}];\n'
                else:
                    c = self.register[op.controls[0]
                                      ].constituent_register.index
                    op_str += f'q[{c}],q[{t}];\n'

            qasm_str += op_str

        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' \
            + self.custom_gates \
            + f'\nqreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\n' \
            + qasm_str

        return qasm_str

    def stim(self) -> str:
        """
        Convert circuit to a string that can be imported by stim.

        Returns
        -------
        stim_str : str
            A string suitable for importing by stim.

        """
        if not self._layout_map:
            self.map_to_physical_layout()
        stim_str = ''

        indent = ''
        for tp in self.timepoints:
            if tp.repeat_start:
                indent = '    '
                stim_str += f'REPEAT {tp.repeat_repetitions}' + ' {\n'
            for op in tp:
                t = self.register[op.targets[0]].constituent_register.index
                if not op.is_controlled:
                    stim_str += indent + f'{op.name} {t}\n'
                else:
                    c = self.register[op.controls[0]
                                      ].constituent_register.index
                    stim_str += indent + f'{op.name} {c} {t}\n'
            if tp.repeat_end:
                indent = ''
                stim_str += '}\n'

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

        lm_dict = dict()
        for item in self._layout_map:
            lm_dict[item[0]] = item[1]
        for op in self:
            if op.name in validops:
                L = [1 for i in range(self.num_qubits)]
                target_qubit = lm_dict[op.targets[0]]
                if not op.is_controlled:
                    L[target_qubit] = op.draw_str_target
                else:
                    control_qubit = lm_dict[op.controls[0]]
                    L[control_qubit] = "•"
                    L[target_qubit] = op.draw_str_target

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

        cur_circ = []

        for ind, op in enumerate(self):
            if ((op.name == 'TICK' and incremental)
                    or ind == len(self)-1):
                cur_circ.append(op)
                cur_circ.append(["id", n-1])
                qc = QuantumCircuit.from_qasm_str(self.qasm())
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
        sample = stim_circ.compile_sampler().sample(1)[0]

        if print_sample:
            print(*(1*sample), sep="")

        if return_sample:
            return 1*sample

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

        if not self._layout_map:
            self.map_to_physical_layout()

        num_qubits = self.num_qubits
        lm = self._layout_map.copy()
        lm.sort(key=lambda x: x[1])
        address_label_len = max(map(len, map(lambda x: str(x[0]), lm)))
        index_label_len = 3 + len(str(num_qubits))
        label_len = address_label_len + index_label_len
        circ_disp = [list(str(lm[i][0]).ljust(address_label_len)
                          + (' : ' + str(lm[i][1])).rjust(index_label_len)
                     + space) for i in range(num_qubits)]
        circ_disp2 = [list(space*(label_len+1))
                      for _ in range(num_qubits)]

        circ_tp_line = [space*(label_len+1)]

        for tp in self.timepoints:
            slices = [[]]
            slices_touched_qubits = [set()]
            for op in tp.operations:

                t = self.register[op.targets[0]].constituent_register.index

                if not op.is_controlled:
                    touched_by_op = set([t])
                else:
                    c = self.register[op.controls[0]
                                      ].constituent_register.index
                    touched_by_op = set(list(range(c, t))
                                        + list(range(t, c)))

                for s in range(len(slices)):
                    if touched_by_op.isdisjoint(slices_touched_qubits[s]):
                        slices[s].append(op)
                        slices_touched_qubits[s].update(touched_by_op)
                        break
                else:
                    slices.append([op])
                    slices_touched_qubits.append(touched_by_op)

            circ_tp_line.append('⍿' + space*(3*(len(slices)-1)+2))

            for sl in slices:
                touched_places = []

                for op in sl:
                    t = self.register[op.targets[0]].constituent_register.index

                    if not op.is_controlled:
                        s = dash + op.draw_str_target + dash
                        circ_disp[t].append(s)
                        circ_disp2[t].append(space*3)
                        touched_places.append(t)

                    elif op.is_controlled:
                        c = self.register[op.controls[0]
                                          ].constituent_register.index
                        vert_places = list(range(c, t)) + list(range(t, c))
                        for i in range(num_qubits):
                            if i == c:
                                circ_disp[i].append(
                                    dash + op.draw_str_control + dash)
                                if i == vert_places[0]:
                                    circ_disp2[i].append(space + vert + space)
                                else:
                                    circ_disp2[i].append(space*3)
                                touched_places.append(i)
                            elif i == t:
                                circ_disp[i].append(
                                    dash + op.draw_str_target + dash)
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

        if filename is None:
            file = sys.stdout
        else:
            file = open(filename, 'w')

        print(''.join(circ_tp_line), file=file, flush=True)
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
        if not self._layout_map:
            self.map_to_physical_layout()

        T = 30
        el = []
        wirey = [20 + i*40 for i in range(self.num_qubits)]

        num_qubits = self.num_qubits
        lm = self._layout_map.copy()
        lm.sort(key=lambda x: x[1])
        address_label_len = max(map(len, map(lambda x: str(x[0]), lm)))
        index_label_len = 3 + len(str(num_qubits))
        labels = [str(lm[i][0]).ljust(address_label_len)
                  + (' : ' + str(lm[i][1])).rjust(index_label_len)
                  for i in range(num_qubits)]
        x0 = max(map(len, labels))*6
        for i in range(num_qubits):
            el.append(svg.Text(x=0, y=wirey[i]+4.5,
                               text=labels[i],
                               class_=["labeltext"]))

        time = 0
        recs = []
        highlight_class = "tp_highlight1"
        for tp in self.timepoints:
            highlight_class = "tp_highlight1" \
                if highlight_class == "tp_highlight2" else "tp_highlight2"
            start_time = time

            slices = [[]]
            slices_touched_qubits = [set()]
            for op in tp.operations:

                t = self.register[op.targets[0]].constituent_register.index

                if not op.is_controlled:
                    touched_by_op = set([t])
                else:
                    c = self.register[op.controls[0]
                                      ].constituent_register.index
                    touched_by_op = set(list(range(c, t))
                                        + list(range(t, c)))

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
                    t = self.register[op.targets[0]].constituent_register.index

                    if not op.is_controlled:

                        el += [
                            svg.Rect(
                                x=x0+time*T+7, y=wirey[t]-10,
                                width=20, height=20,
                                class_=["gaterect"]
                            ),
                            svg.Text(
                                x=x0+time*T+10, y=wirey[t] + 7.75,
                                text=op.name[-1],
                                class_=["gatetext"]
                            )]

                    elif op.is_controlled:
                        c = self.register[op.controls[0]
                                          ].constituent_register.index

                        el += [
                            svg.Circle(
                                cx=x0+time*T+17, cy=wirey[c], r=3,
                                class_=["control1"]
                            ),
                            svg.Line(
                                x1=x0+time*T+17, x2=x0+time*T+17,
                                y1=wirey[c]+3, y2=wirey[t]-10,
                                class_=["controlline"]
                            ),
                            svg.Rect(
                                x=x0+time*T+7, y=wirey[t]-10,
                                width=20, height=20,
                                class_=["gaterect"]
                            ),
                            svg.Text(
                                x=x0+time*T+10, y=wirey[t]+7.75,
                                text=op.name[-1],
                                class_=["gatetext"]
                            )
                        ]
                time += 1
            if highlight_timepoints:
                recs.append(svg.Rect(
                    x=x0+start_time*T+2.25, y=0,
                    width=(time-start_time)*T, height=wirey[-1]+20,
                    class_=[highlight_class]))

        el = [svg.Style(
            text="""
            .labeltext { font-size: 12px; font-weight: 400; fill: black;}
            .qubitline { stroke: black; stroke-width: 2; }
            .gatetext { font: 20px sans-serif; font-weight: 400; fill: black;}
            .gaterect { fill: white; stroke: black; stroke-width: 2 }
            .control1 { fill: black; stroke: black; stroke-width: 1 }
            .controlline { stroke: black; stroke-width: 2}
            .tp_highlight1 { fill: red; opacity: 0.2;}
            .tp_highlight2 { fill: blue; opacity: 0.2;}
                """)] + recs + \
            [svg.Line(
                x1=x0+0, x2=x0+time*T+30,
                y1=y, y2=y,
                class_=["qubitline"]) for y in wirey] + el

        s = svg.SVG(
            width=x0+time*T+2.25,
            height=wirey[-1]+19,
            elements=el,
        )

        if filename is None:
            display(SVG(s.as_str()))
        else:
            with open(filename, 'w') as f:
                f.write(s.as_str())
