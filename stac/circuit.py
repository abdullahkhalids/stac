"""Provide a module to create and manipulate quantum circuits."""
from IPython.display import display
import json
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
import stim
import copy
import tabulate
tabulate.PRESERVE_WHITESPACE = True


_single_qubit_gates = {'X', 'Y', 'Z', 'H'}
_multi_qubit_gates = {'CX', 'CY', 'CZ'}
_measurements = {'R', 'M', 'MR'}
_circuit_annotations = {'TICK'}

_quantum_operations = set.union(
    _single_qubit_gates,
    _multi_qubit_gates,
    _measurements,
)

_circuit_operations = set.union(
    _quantum_operations,
    _circuit_annotations
)


def display_states(head, *args):
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

    error_ind = 0

    def __init__(self, *args):
        """Construct a quantum circuit."""
        self.circuit = []
        self.num_qubits = 0
        self.custom_gates = ''

    def __str__(self):
        """Class description."""
        return self.__repr__()

    def __repr__(self):
        """Circuit description."""
        str_circ = ''
        for op in self.circuit:
            for item in op[1:]:
                str_circ += f'{item} '
            str_circ += '\n'
        return str_circ

    def __iter__(self):
        """Return iterator for the quantum circuit."""
        return self.circuit.__iter__()

    def __len__(self):
        """Return number of operations in circuit."""
        return self.circuit.__len__()

    def append(self, *args):
        """
        Append an operation or annotation to the circuit.

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Legacy. If we pass in a list in the format
        # ['op', qubits] then add to circuit as is
        if (len(args) == 1
                and type(args[0]) is str
                and args[0].upper() == 'TICK'):
            op = list(args) + [0, self.num_qubits-1]
        elif len(args) == 1 and type(args[0]) is list:
            op = args[0]
        elif len(args) == 2 or len(args) == 3:
            op = list(args)
        else:
            raise Exception("Can't recognize input.")

        # check if valid operation
        if (type(op[0]) is not str
                or any(type(item) is not int for item in op[1:])):
            raise Exception("Can't recognize input.")

        if op[0].upper() in _quantum_operations:
            op.insert(0, 'Q')
        elif op[0].upper() in _circuit_annotations:
            op.insert(0, 'A')
        else:
            op.insert(0, 'U')
            # raise Exception(f'{op[0]} is not a valid operation')

        op[1] = op[1].upper()

        # increment number of qubits if needed
        for q in op[2:]:
            if q+1 > self.num_qubits:
                self.num_qubits = q+1

        self.circuit.append(op)

    def __add__(self, circuit2):
        """
        Compose two circuits.

        Parameters
        ----------
        circuit2 : Circuit
            The circuit to be added to this one.

        Returns
        -------
        new_circuit : Circuit
            The composition of the two circuits.

        """
        new_circuit = Circuit()
        new_circuit.circuit += self.circuit
        new_circuit.circuit += circuit2.circuit

        new_circuit.num_qubits = max(self.num_qubits, circuit2.num_qubits)

        new_circuit.custom_gates = (self.custom_gates
                                    + '\n'
                                    + circuit2.custom_gates)
        return new_circuit

    def __getitem__(self, ind):
        """Make circuit subscriptable."""
        return self.circuit.__getitem__(ind)

    def compose(self, circuit2, *args):
        """
        Compose circuit2 to this circuit with first qubit at index start_ind.

        Parameters
        ----------
        circuit2 : Circuit
            Circuit that will be composed with this circuit.
        *args : int or List
            int: Index of where the first qubit of circuit2 will be placed.
            List: Should have as many elements as circuit2.num_qubits. The ith
            entry specifies where the ith qubit will go.

        Returns
        -------
        None.

        """
        qubit_map = dict()
        if type(args[0]) is int:
            for i in range(circuit2.num_qubits):
                qubit_map[i] = i + args[0]
            if circuit2.num_qubits + args[0] + 1 > self.num_qubits:
                self.num_qubits = circuit2.num_qubits + args[0] + 1

        elif type(args[0]) is list:
            if len(args[0]) != circuit2.num_qubits:
                raise ValueError

            for i in range(circuit2.num_qubits):
                qubit_map[i] = args[0][i]
                if args[0][i] + 1 > self.num_qubits:
                    self.num_qubits = args[0][i] + 1
        else:
            raise ValueError

        for op in circuit2:
            shifted_op = copy.deepcopy(op)
            shifted_op[2] = qubit_map[shifted_op[2]]
            if len(op) == 4:
                shifted_op[3] = qubit_map[shifted_op[3]]
            self.circuit.append(shifted_op)

        self.custom_gates = (self.custom_gates
                             + '\n'
                             + circuit2.custom_gates)

    def _next_error(self, q_ind):
        self.custom_gates += f'gate e{Circuit.error_ind} a {{id a;}}\n'
        er_op = ['E', 'e'+str(Circuit.error_ind), q_ind]
        Circuit.error_ind += 1
        return er_op

    def append_error(self, indices):
        """
        Append a numbered error(s) to the circuit.

        Parameters
        ----------
        indices : int or List
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if type(indices) == int:
            self.circuit.append(self._next_error(indices))
        else:
            for ind in indices:
                self.circuit.append(self._next_error(ind))

    def insert_errors(self,
                      before_op=None,
                      after_op=None,
                      error_types=None):
        """
        Insert errors after operations of a certain kind.

        Only one of after_op and before_op can be passed.

        Parameters
        ----------
        before_op : str, optional
            The name of the operation after which error is applied.
            eg. "X" or "CX". The default is None.
        after_op : str, optional
            The name of the operation after which error is applied.
            eg. "X" or "CX". The default is None
        error_type : List[str]
            If after_op/before_op is a single qubit operation, must be a
            length one list that is the name of single-qubit gate.
            If after_op/before_op is a two qubit operation, must be a list of
            two single-qubit gates. If any gate is $I$, then it won't be
            inserted. The default is None.

        Returns
        -------
        None.

        """
        if before_op is not None and after_op is not None:
            raise Exception("Can only pass one of before_op and after_op.")

        def create_error_operator():
            if len(op) == 3:
                if error_types[0] == "E":
                    er_op = [self._next_error(op[2])]
                elif error_types[0] != "I":
                    er_op = [['E', error_types[0], op[2]]]
            else:
                er_op = []
                if error_types[0] == "E":
                    er_op += [self._next_error(op[2])]
                elif error_types[0] != "I":
                    er_op += [['E', error_types[0], op[2]]]

                if error_types[1] == "E":
                    er_op += [self._next_error(op[3])]
                elif error_types[1] != "I":
                    er_op += [['E', error_types[1], op[3]]]

            return er_op

        error_circuit = []
        if after_op == "start":
            for i in range(self.num_qubits):
                op = ['Q', '', i]
                error_circuit.extend(create_error_operator())
            error_circuit.extend(self.circuit)
            self.circuit = error_circuit
            return
        elif before_op == "end":
            error_circuit.extend(self.circuit)
            for i in range(self.num_qubits):
                op = ['Q', '', i]
                error_circuit.extend(create_error_operator())
            self.circuit = error_circuit
            return

        for op in self.circuit:
            if before_op is not None and op[1] == before_op:
                error_circuit.extend(create_error_operator())

            error_circuit.append(op)

            if after_op is not None and op[1] == after_op:
                error_circuit.extend(create_error_operator())

        self.circuit = error_circuit

    def without_noise(self):
        """
        Return a version of the circuit with no errors.

        Returns
        -------
        noiseless_circuit : Circuit
            Circuit with no error operations.

        """
        noiseless_circuit = Circuit()
        for op in self.circuit:
            if op[0] != 'E':
                noiseless_circuit.append(op[1:])
        return noiseless_circuit

    def qasm(self):
        """
        Convert circuit to qasm string.

        Parameters
        ----------
        custom_gates : str, optional
            Qasm style definition of custom gates. The default is ''.

        Returns
        -------
        qasm_str : str
            The qasm string of the circuit.

        """
        qasm_str = ''

        for op in self.circuit:
            # op[0] is the gate name
            if op[1][:7] == 'X_ERROR':
                continue
            elif op[1] == 'R':
                op_str = f'reset q[{op[2]}];\n'
            elif op[1] == 'MR' or op[1] == 'M':
                op_str = f'measure q[{op[2]}] -> c[{op[2]}];\n'
            elif op[1] == 'I':
                op_str = f'id q[{op[2]}];\n'
            elif op[1].upper() == 'TICK':
                op_str = 'barrier '
                for i in range(op[2], op[3]+1):
                    op_str += f'q[{i}],'
                op_str = op_str[:-1] + ';\n'
            else:
                op_str = op[1].lower() + ' '
                # followed by one or two arguments
                if len(op) == 3:
                    op_str += f'q[{op[2]}];\n'
                else:
                    op_str += f'q[{op[2]}],q[{op[3]}];\n'

            qasm_str += op_str

        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' \
            + self.custom_gates \
            + f'\nqreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\n' \
            + qasm_str

        return qasm_str

    def stim(self):
        """
        Convert circuit to a string that can be imported by stim.

        Returns
        -------
        stim_str : str
            A string suitable for importing by stim.

        """
        stim_str = ''

        for op in self.circuit:

            if op[0] == 'E' and op[1][0] == 'e':
                continue

            # op[1] is the gate name
            op_str = op[1] + ' '

            if op[1] == "TICK":
                op_str += '\n'
            elif len(op) == 3:
                op_str += f'{op[2]} \n'
            else:
                op_str += f'{op[2]} {op[3]}\n'

            stim_str += op_str

        return stim_str

    def quirk(self):
        """
        Convert circuit to a quirk circuit.

        Returns
        -------
        None.
        Prints a url that can opened in the browser.
        """
        validops = {'H', 'X', 'Y', 'Z', 'CX', 'CY', 'CZ'}
        cols = []
        for op in self.circuit:
            if op[1] in validops:
                L = [1 for i in range(self.num_qubits)]
                if len(op) == 3:
                    L[op[2]] = op[1]
                else:
                    L[op[2]] = "•"
                    L[op[3]] = op[1][1]

                cols.append(L)

        url = 'https://algassert.com/quirk#circuit={"cols":' + \
            json.dumps(cols, ensure_ascii=False) + '}'

        print(url)

    def draw(self, **kwargs):
        """
        Draw the circuit using Qiskit.

        Parameters
        ----------
        custom_gates : str, optional
            Qasm style description of any special gates. The default is ''.
        **kwargs : any
            Any addtional arguments are sent to the qiskit function.

        Returns
        -------
        None.

        """
        if len(kwargs) == 0:
            display(
                QuantumCircuit.from_qasm_str(
                    self.qasm()
                ).draw(output='latex')
            )
        else:
            display(
                QuantumCircuit.from_qasm_str(
                    self.qasm()
                ).draw(**kwargs)
            )

    def simulate(self,
                 head=None,
                 incremental=False,
                 return_state=False,
                 print_state=True):
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

        for ind in range(len(self.circuit)):
            op = self.circuit[ind]
            if ((op[0].upper() == 'TICK' and incremental)
                    or ind == len(self.circuit)-1):
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
               return_sample=False,
               print_sample=True):
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
        TYPE
            DESCRIPTION.

        """
        stim_circ = stim.Circuit(self.stim())
        print(stim_circ)
        print("\n\n\n")
        sample = stim_circ.compile_sampler().sample(1)[0]

        if print_sample:
            print(*(1*sample), sep="")

        if return_sample:
            return 1*sample

    def simulate_errors(self, library, error_gate):
        """
        Show the effect of each error in an annotated circuit.

        Given a circuit annotated with errors, for each error simulate the
        circuit with the error replaced by the error_gate.

        Parameters
        ----------
        library: str
            'qiskit' or 'stim'
        error_gate : str
            Any single-qubit pauli accepted by Qiskit.

        Returns
        -------
        None.

        """
        if library == 'qiskit':
            self._simulate_errors_qiskit(error_gate)
        elif library == 'stim':
            self._simulate_errors_stim(error_gate)

    def _simulate_errors_qiskit(self, error_gate):
        """
        Simulate an error circuit using qiskit.

        Parameters
        ----------
        error_gate : str
            Any single-qubit pauli accepted by Qiskit.

        Returns
        -------
        None.

        """
        state_tabs = []
        # first simulate the no, error state
        state_tabs.append(self.simulate(print_state=False, return_state=True))
        head = ['basis', 'no error']
        # then iterate through the circuit
        for i in range(len(self)):
            if self.circuit[i][0] != 'E':
                continue
            # if op is an error, then copy the circuit and simulate
            circ_copy = Circuit()
            circ_copy.circuit = self.circuit.copy()
            circ_copy.num_qubits = self.num_qubits
            circ_copy.custom_gates = self.custom_gates
            circ_copy.circuit.insert(i+1,
                                     ['Q', error_gate, self.circuit[i][-1]])
            state_tabs.append(circ_copy.simulate(
                print_state=False,
                return_state=True))
            head.append(self.circuit[i][1])

        display_states(head, *state_tabs)

    def _sample_error_stim(self, error_gate):

        tab = []
        head = ['error', 'syndrome']
        for i in range(len(self)):
            if self.circuit[i][0] != 'E':
                continue
            # if op is an error, then copy the circuit and simulate
            circ_copy = Circuit()
            circ_copy.circuit = self.circuit.copy()
            circ_copy.num_qubits = self.num_qubits
            circ_copy.custom_gates = self.custom_gates
            circ_copy.circuit.insert(i+1,
                                     ['Q', error_gate, self.circuit[i][-1]])

            tab.append([self.circuit[i][1],
                       circ_copy.sample(print_sample=False,
                                        return_sample=True)])

        print(tabulate.tabulate(tab, headers=head, colalign=None))
