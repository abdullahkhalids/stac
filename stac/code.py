"""Stac is a stabilizer code module."""

from itertools import combinations
from IPython.display import display, Math
import numpy as np

# from .operation import Operation
from .timepoint import Timepoint
# from .qubit import PhysicalQubit, VirtualQubit
from .register import QubitRegister, RegisterRegister
from .circuit import Circuit
from .supportedoperations import _quantum_operations


def print_matrix(array, augmented=False):
    """
    Display an array using latex.

    If augmented=True, then a line is placed
    in the center of the matrix, which is useful
    for printing the stabilizer generator matrix.
    """
    # If array is a list
    if array.ndim == 1:
        array = np.array([array])

    data = ''
    for line in array:
        for element in line:
            data += str(element) + ' & '
        data = data[:-3]
        data += r' \\' + '\n'

    if augmented:
        matname = '{array}'
        (nrows, ncols) = array.shape
        n = int(ncols/2)
        c = ''.join(['c' for i in range(n)])
        colalign = '{' + c + '|' + c + '}'
        display(Math(
            f'''\
                \\left(\\begin{matname}{colalign}
                       {data}
                       \\end{matname}\\right)\
                    '''))
    else:
        matname = '{pmatrix}'
        display(Math(f'\\begin{matname}\n {data}\\end{matname}'))


def print_paulis(G):
    """Print a set of Paulis as I,X,Y,Z."""
    if G.ndim == 1:
        m = 1
        n = int(G.shape[0]/2)
        G = np.array([G])
    else:
        m = G.shape[0]
        n = int(G.shape[1]/2)

    for i in range(m):
        pauli_str = ''
        for j in range(n):
            if G[i, j] == 0 and G[i, n+j] == 0:
                pauli_str += 'I'
            elif G[i, j] and G[i, n+j]:
                pauli_str += 'Y'
            elif G[i, j]:
                pauli_str += 'X'
            elif G[i, n+j]:
                pauli_str += 'Z'

        display(Math(f'${pauli_str}$'))


def print_paulis_indexed(G):
    """Print a set of Paulis as indexed X,Y,Z."""
    if len(G.shape) == 1:
        m = 1
        n = int(G.shape[0]/2)
        G = np.array([G])
    else:
        m = G.shape[0]
        n = int(G.shape[1]/2)

    for i in range(m):
        pauli_str = ''
        for j in range(n):
            if G[i, j] == 0 and G[i, n+j] == 0:
                pass
            elif G[i, j] and G[i, n+j]:
                pauli_str += 'Y_{{ {0} }}'.format(j)
            elif G[i, j]:
                pauli_str += 'X_{{ {} }}'.format(j)
            elif G[i, n+j]:
                pauli_str += 'Z_{{ {} }}'.format(j)

        if pauli_str != '':
            display(Math(f'${pauli_str}$'))


def _rref(A, colswap=True):
    """
    Operations to row reduce.

    Determine the set of elementary operations
    that give the reduced row echelon form (RREF) of A
    Returns the matrix rank, reduced matrix,
    and operations.
    """
    M = np.copy(A)
    (nrow, ncol) = M.shape

    if nrow == 0 or ncol == 0:
        return M, 0, []

    ops = []
    cur_col = 0

    # iterate over each row and find pivot
    for cur_row in range(nrow):

        # determine the first non-zero col
        new_row = cur_row
        new_col = cur_col
        while M[new_row, new_col] == 0:
            new_row += 1
            if new_row == nrow:
                new_row = cur_row
                new_col += 1
                # if rest of matrix is zero
                if new_col == ncol:
                    return M, cur_row, ops

        # the first non-zero entry is M[cur_row, new_col]
        # swap cols to bring it forward
        if cur_col != new_col and colswap:
            M[:, [cur_col, new_col]] = M[:, [new_col, cur_col]]
            ops.append(["colswap", cur_col, new_col])

        # Move it to the top
        if new_row != cur_row:
            M[[cur_row, new_row], :] = M[[new_row, cur_row], :]
            ops.append(["rowswap", cur_row, new_row])

        # now non-zero entry is at M[r, cur_col]

        # place zeros above and below the pivot position
        for r in range(nrow):
            if r != cur_row and M[r, cur_col]:
                M[r, :] = (M[r, :] + M[cur_row, :]) % 2
                ops.append(["addrow", r, cur_row])

        cur_col += 1

        # if we are are done with all cols
        if cur_col == ncol:
            break

    # rank is how far down we have gone
    rank = cur_row+1

    return M, rank, ops


def _perform_row_operations(A, ops, start_row=0):
    """
    Perform row operations on a matrix.

    Apply a set of elementary row operations, ops,
    to matrix A. If start_row is specified, then
    operations are performed relative to it.
    """
    M = np.copy(A)

    for op in ops:
        if op[0] == "colswap":
            M[:, [op[1], op[2]]] = M[:, [op[2], op[1]]]
        elif op[0] == "rowswap":
            M[[start_row + op[1], start_row + op[2]], :] = \
                M[[start_row + op[2], start_row + op[1]], :]
        elif op[0] == "addrow":
            M[start_row + op[1], :] = (M[start_row + op[1], :]
                                       + M[start_row + op[2], :]) % 2

    return M


def _inner_product(v, w):

    n = int(len(v)/2)
    return (v[:n]@w[n:] + v[n:]@w[:n]) % 2


class Code:
    """Class for creating stabilizer codes."""

    def __init__(self, *args):
        """Construct a stabilizer code."""
        if len(args) == 1:
            self.generator_matrix = args[0]

            self.num_physical_qubits = int(self.generator_matrix.shape[1]/2)

            self.generators_x = \
                self.generator_matrix[:, :self.num_physical_qubits]
            self.generators_z = \
                self.generator_matrix[:, self.num_physical_qubits:]

        elif len(args) == 2:
            if args[0].shape != args[1].shape:
                print("The shape of the matrices don't match")
                return

            self.generators_x = args[0]
            self.generators_z = args[1]

            self.generator_matrix = np.concatenate(
                (self.generators_x, self.generators_z), axis=1)

            self.num_physical_qubits = self.generators_x.shape[1]

        self.num_generators = self.generators_x.shape[0]

        self.num_logical_qubits = self.num_physical_qubits \
            - self.num_generators

        self.distance = None

        self.rankx = None

        self.standard_generators_x = None
        self.standard_generators_z = None
        self.standard_generator_matrix = None

        self.destab_gen_mat = None

        self.logical_xs = None
        self.logical_zs = None
        self.logical_circuits = dict()
        for op in _quantum_operations:
            self.logical_circuits[op] = None

        self.encoding_circuit = None
        self.decoding_circuit = None

    def __repr__(self):
        """Return way to make code."""
        pass
        return f'Code(\n{self.generator_matrix}\n)'

    def __str__(self):
        """Return description of code."""
        return 'A [[{},{}]] code'.format(self.num_physical_qubits,
                                         self.num_logical_qubits)

    def check_valid_code(self):
        """Check if code generators commute."""
        is_valid = True
        for i in range(self.num_generators-1):
            for j in range(i+1, self.num_generators):
                if (self.generators_x[i]@self.generators_z[j]
                        + self.generators_x[j]@self.generators_z[i]) % 2:
                    print("Generators {} and {} don't commute".format(i, j))
                    print(np.append(self.generators_x[i],
                                    self.generators_z[i]))
                    print(np.append(self.generators_x[j],
                                    self.generators_z[j]))
                    is_valid = False

        return is_valid

    def construct_standard_form(self):
        """
        Construct the standard form a stabilizer matrix.

        Returns
        -------
        standard_generators_x
            The X part of the standard generator matrix.
        standard_generators_z
            The Z part of a standard generator matix.
        rankx
            The rank of the X part of the generator matrix..

        """
        # Find RREF of X stabs
        (standard_generators_x, self.rankx, opsA) = _rref(self.generators_x)

        standard_generators_z = _perform_row_operations(self.generators_z,
                                                        opsA)

        # now extract E' and reduce
        (rEp, rankEp, opsEp) = _rref(standard_generators_z[self.rankx:,
                                                           self.rankx:])

        # perform same operations on full matrices
        self.standard_generators_x = _perform_row_operations(
            standard_generators_x,
            opsEp,
            self.rankx)

        self.standard_generators_z = _perform_row_operations(
            standard_generators_z,
            opsEp,
            self.rankx)

        self.standard_generator_matrix = np.concatenate(
            (self.standard_generators_x,
             self.standard_generators_z),
            axis=1)

        return self.standard_generators_x,\
            self.standard_generators_z,\
            self.rankx

    def construct_logical_operators(self):
        """
        Construct logical operators for the code.

        Uses Gottesman's method for doing this.

        Returns
        -------
        logical_xs
            Array of logical xs. Each row is an operator.
        logical_zs
            Array of logical xs. Each row is an operator.
        """
        if self.standard_generators_x is None:
            self.construct_standard_form()

        n = self.num_physical_qubits
        k = self.num_logical_qubits
        r = self.rankx

        # The relevant parts of the reduced generator matrix are
        A2 = self.standard_generators_x[0:r, (n-k):n]
        C1 = self.standard_generators_z[0:r, r:(n-k)]
        C2 = self.standard_generators_z[0:r, (n-k):n]
        E = self.standard_generators_z[r:(n-k), (n-k):n]

        # Construct the logical X operators
        self.logical_xs = np.concatenate((
            np.zeros((k, r), dtype=int),
            E.transpose(),
            np.identity(k, dtype=int),
            (E.transpose()@C1.transpose() + C2.transpose()) % 2,
            np.zeros((k, n-r), dtype=int)
        ), axis=1, dtype=int)

        # Construct the logical Z operators
        self.logical_zs = np.concatenate((
            np.zeros((k, n), dtype=int),
            A2.transpose(),
            np.zeros((k, n-k-r), dtype=int),
            np.identity(k, dtype=int)
        ), axis=1, dtype=int)

        return self.logical_xs, self.logical_zs

    def construct_logical_gate_circuits(self,
                                        syndrome_measurement_type='non_ft'):
        """
        Create the circuits that implement logical circuits for the code.

        Results are storted in logical_circuits.

        Returns
        -------
        None.

        """
        if self.logical_xs is None:
            self.construct_logical_operators()

        n = self.num_physical_qubits

        for name, operators in [('X', self.logical_xs),
                                ('Z', self.logical_zs)]:
            circ = Circuit()
            circ.append_register(
                self.construct_encoded_qubit_register(
                    0,
                    syndrome_measurement_type))

            for pauli in operators:
                for i in range(n):
                    if pauli[i] and pauli[n+i]:
                        circ.append('Y', (0, 0, 0, i))
                    elif pauli[i]:
                        circ.append('X', (0, 0, 0, i))
                    elif pauli[n+i]:
                        circ.append('Z', (0, 0, 0, i))

            self.logical_circuits[name] = circ

        circ = Circuit()
        circ.append_register(
            self.construct_encoded_qubit_register(
                0, syndrome_measurement_type))
        for i in range(n):
            circ.append('H', (0, 0, 0, i))
        self.logical_circuits['H'] = circ

        for name in ['M', 'R', 'MR']:
            circ = Circuit()
            circ.append_register(
                self.construct_encoded_qubit_register(
                    0, syndrome_measurement_type))
            for i in range(self.num_physical_qubits):
                circ.append(name, (0, 0, 0, i))
            self.logical_circuits[name] = circ

        for name in ['CX', 'CZ']:
            circ = Circuit()
            circ.append_register(
                self.construct_encoded_qubit_register(
                    0, syndrome_measurement_type))
            circ.append_register(
                self.construct_encoded_qubit_register(
                    0, syndrome_measurement_type))
            for i in range(n):
                circ.append(name, (0, 0, 0, i), (0, 1, 0, i))

            self.logical_circuits[name] = circ

    def find_destabilizers(self):
        """
        Find the destabilizers of the standard form generators.

        Find the destabilizers of the standard form generators by exhaustive
        search. This will be slow for large codes but has the advantage that
        it will find the lowest weight destabilizers.

        Returns
        -------
        destab_gen_mat: numpy.array
            Array of shape m x 2n where each row is a destabilizer
        TYPE
            DESCRIPTION.

        """
        if self.standard_generators_x is None:
            self.construct_standard_form()

        n = self.num_physical_qubits

        destabs = np.empty((self.num_generators, 2*n), dtype=int)
        destab_found = [False for i in range(self.num_generators)]
        i = 0
        b = False

        # these for loops create binary vector arrays
        # in increasing number of 1s.
        for k in range(1, 2*n):
            if b:
                break
            for comb in combinations(np.arange(2*n), k):
                v = np.array([1 if i in comb else 0 for i in range(2*n)])

                # v should anti-commute with only one generator
                # to be a destabilizer
                num_anti_commute = 0
                for i in range(self.num_generators):
                    ip = _inner_product(self.standard_generator_matrix[i], v)
                    if ip:
                        num_anti_commute += ip
                        if num_anti_commute > 1:
                            break
                        else:
                            destab_for_gen = i
                else:
                    if not destab_found[destab_for_gen]:
                        destabs[destab_for_gen] = v
                        destab_found[destab_for_gen] = True

                        if np.all(destab_found):
                            b = True
                            break

        self.destab_gen_mat = np.array(destabs)
        return self.destab_gen_mat

    def construct_syndrome_measurement_register(self,
                                                level,
                                                syndrome_measurement_type):
        """
        Create a register appropriate for doing syndrome measurements for
        qubits encoded by this code.

        Parameters
        ----------
        level : int
            The concatenation level of the qubit.
        syndrome_measurement_type : str
            Options are 'non_ft',
                        'cat',
                        'cat_standard'.
            With the 'standard' postfix uses the standard form of the
            generators. If no argument, then 'non_ft' is Default.

        Returns
        -------
        RegisterRegister
            Register for encoded qubit.
        """
        if syndrome_measurement_type == 'cat_standard':
            genmat = self.standard_generator_matrix
        else:
            genmat = self.generator_matrix

        if syndrome_measurement_type == 'non_ft':
            genregs = [QubitRegister('g', level, 1)
                       for i in range(self.num_generators)]
        elif syndrome_measurement_type in ['cat', 'cat_standard']:
            genregs = []
            for g in genmat:
                cat_reg = QubitRegister('c', level, sum(g))
                detect_reg = QubitRegister('f', level, 1)
                genregs.append(
                    RegisterRegister('g',
                                     level,
                                     subregisters=(cat_reg, detect_reg)))
        else:
            raise Exception("Unknown 'syndrome_measurement_type'")
        return RegisterRegister('s', level, genregs, code=self)

    def construct_encoded_qubit_register(self,
                                         level,
                                         syndrome_measurement_type):
        """
        Create a register appropriate for creating an encoded qubit.

        Parameters
        ----------
        level : int
            The concatenation level of the qubit.
        syndrome_measurement_type : str
            Options are 'non_ft',
                        'non_ft_standard',
                        'cat',
                        'cat_standard'.
            With the 'standard' postfix uses the standard form of the
            generators. If no argument, then 'non_ft' is Default.

        Returns
        -------
        RegisterRegister
            Register for encoded qubit.

        """
        n = self.num_physical_qubits
        datareg = QubitRegister('d', level, n)
        if syndrome_measurement_type == 'non_ft':
            genregs = [QubitRegister('g', level, 1)
                       for i in range(self.num_generators)]
        elif syndrome_measurement_type == 'cat':
            genregs = []
            for g in self.generator_matrix:
                cat_reg = QubitRegister('c', level, sum(g))
                detect_reg = QubitRegister('a', level, 1)
                genregs.append(
                    RegisterRegister('g',
                                     level,
                                     subregisters=(cat_reg, detect_reg)))
        else:
            raise Exception("Unknown 'syndrome_measurement_type'")
        syndreg = RegisterRegister('s', level, genregs)
        return RegisterRegister('e',
                                level,
                                subregisters=(datareg, syndreg),
                                code=self)

    def construct_encoding_circuit(self):
        """
        Construct an encoding circuit for the code using Gottesman's method.

        Returns
        -------
        encoding_circuit : Circuit
            The encoding circuit.

        """
        if self.logical_xs is None:
            self.construct_logical_operators()

        n = self.num_physical_qubits
        k = self.num_logical_qubits
        r = self.rankx

        self.encoding_circuit = Circuit()
        # rg = QubitRegister('d', 0, n)
        # self.encoding_circuit.append_register(rg)
        self.encoding_circuit.append_register(
            self.construct_encoded_qubit_register(0, 'non_ft'))
        self.encoding_circuit.base_address = (0, 0, 0)
        for i in range(k):
            for j in range(r, n-k):
                if self.logical_xs[i, j]:
                    self.encoding_circuit.append("CX", n-k+i, j)

        for i in range(r):
            self.encoding_circuit.append("H", i)
            for j in range(n):
                if i == j:
                    continue
                if (self.standard_generators_x[i, j]
                        and self.standard_generators_z[i, j]):

                    self.encoding_circuit.append("CX", i, j)
                    self.encoding_circuit.append("CZ", i, j)
                elif self.standard_generators_x[i, j]:
                    self.encoding_circuit.append("CX", i, j)
                elif self.standard_generators_z[i, j]:
                    self.encoding_circuit.append("CZ", i, j)

        return self.encoding_circuit

    def construct_decoding_circuit(self):
        """
        Construct an decoding circuit for the code using Gottesman's method.

        Returns
        -------
        decoding_circuit : Circuit
            The decoding circuit.

        """
        if self.logical_xs is None:
            self.construct_logical_operators()

        n = self.num_physical_qubits

        self.decoding_circuit = Circuit()

        # Note, we will need num_logical_qubits ancilla
        for i in range(len(self.logical_zs)):
            for j in range(n):
                if self.logical_zs[i, n+j]:
                    self.decoding_circuit.append(["CX", j, n+i])

        for i in range(len(self.logical_xs)):
            for j in range(n):
                if self.logical_xs[i, j] and self.logical_xs[i, n+j]:
                    self.decoding_circuit.append(["CZ", n+i, j])
                    self.decoding_circuit.append(["CX", n+i, j])
                elif self.logical_xs[i, j]:
                    self.decoding_circuit.append(["CX", n+i, j])
                elif self.logical_xs[i, n+j]:
                    self.decoding_circuit.append(["CZ", n+i, j])

        return self.decoding_circuit

    def construct_syndrome_circuit(self, *args):
        """
        Construct a circuit to measure the stabilizers of the code.

        ----------
        *args : str
            Options are 'non_ft',
                        'non_ft_standard',
                        'cat',
                        'cat_standard'.
            With the 'standard' postfix uses the standard form of the
            generators. If no argument, then 'non_ft' is Default.

        Returns
        -------
        syndrome_circuit : Circuit
            The circuit for measuring the stabilizers.

        """
        if len(args) == 0:
            self.syndrome_circuit = self._construct_syndrome_circuit_simple(
                self.generators_x, self.generators_z)
        elif type(args[0]) is str:
            if args[0] == 'non_ft':
                self.syndrome_circuit = \
                    self._construct_syndrome_circuit_simple(self.generators_x,
                                                            self.generators_z)
            elif args[0] == 'non_ft_standard':
                if self.standard_generators_x is None:
                    self.construct_standard_form()
                self.syndrome_circuit = \
                    self._construct_syndrome_circuit_simple(
                        self.standard_generators_x,
                        self.standard_generators_z)

            elif args[0] == 'cat':
                self.syndrome_circuit = \
                    self._construct_syndrome_circuit_cat(self.generators_x,
                                                         self.generators_z)
            elif args[0] == 'cat_standard':
                if self.standard_generators_x is None:
                    self.construct_standard_form()
                self.syndrome_circuit = \
                    self._construct_syndrome_circuit_cat(
                        self.standard_generators_x,
                        self.standard_generators_z)

        return self.syndrome_circuit

    def _construct_syndrome_circuit_simple(self, generators_x, generators_z):
        """Construct a non-fault tolerant syndrome circuit."""
        self.syndrome_circuit = Circuit()
        rg = self.construct_encoded_qubit_register(0, 'non_ft')
        self.syndrome_circuit.append_register(rg)

        for i in range(self.num_generators):
            # first apply hadamard to ancilla
            self.syndrome_circuit.append("H", (0, 0, 1, i, 0))

        for i in range(self.num_generators):
            for j in range(self.num_physical_qubits):
                if generators_x[i, j] and generators_z[i, j]:
                    self.syndrome_circuit.append("CX",
                                                 (0, 0, 1, i, 0),
                                                 (0, 0, 0, j))
                    self.syndrome_circuit.append("CZ",
                                                 (0, 0, 1, i, 0),
                                                 (0, 0, 0, j))
                elif generators_x[i, j]:
                    self.syndrome_circuit.append("CX",
                                                 (0, 0, 1, i, 0),
                                                 (0, 0, 0, j))
                elif generators_z[i, j]:
                    self.syndrome_circuit.append("CZ",
                                                 (0, 0, 1, i, 0),
                                                 (0, 0, 0, j))
            self.syndrome_circuit.cur_time += 1

        for i in range(0, self.num_generators):
            # last apply hadamard to ancilla
            self.syndrome_circuit.append("H", (0, 0, 1, i, 0))

        # self.syndrome_circuit.append('MR', (0, 0, 1, 0, 0), time=[1])
        for i in range(0, self.num_generators):
            # change to MR
            self.syndrome_circuit.append('MR', (0, 0, 1, i, 0))

        return self.syndrome_circuit

    def _construct_syndrome_circuit_cat(self, generators_x, generators_z):
        """Construct a fault tolerant syndrome circuit."""
        self.syndrome_circuit = Circuit()
        rg = self.construct_encoded_qubit_register(0, 'cat')
        self.syndrome_circuit.append_register(rg)

        for i in range(self.num_generators):
            self.syndrome_circuit.cur_time = 0
            self.syndrome_circuit.append('H',  (0, 0, 1, i, 0, 0))
            ng = self.syndrome_circuit.register[0, 0, 1, i, 0].num_qubits
            for j in range(ng-1):
                self.syndrome_circuit.append(
                    'CX', (0, 0, 1, i, 0, j), (0, 0, 1, i, 0, j+1))
            self.syndrome_circuit.append(
                'CX', (0, 0, 1, i, 0, 0), (0, 0, 1, i, 1, 0))
            self.syndrome_circuit.append(
                'CX', (0, 0, 1, i, 0, ng-1), (0, 0, 1, i, 1, 0))
            # self.syndrome_circuit.append('MR', (0, 0, 1, i, 1, 0))

        self.syndrome_circuit.cur_time += 1

        for i in range(self.num_generators):
            k = 0
            for j in range(self.num_physical_qubits):
                if generators_x[i, j] and generators_z[i, j]:
                    self.syndrome_circuit.append("CX",
                                                 (0, 0, 1, i, 0, k),
                                                 (0, 0, 0, j))
                    self.syndrome_circuit.append("CZ",
                                                 (0, 0, 1, i, 0, k),
                                                 (0, 0, 0, j))
                    k += 1
                elif generators_x[i, j]:
                    self.syndrome_circuit.append("CX",
                                                 (0, 0, 1, i, 0, k),
                                                 (0, 0, 0, j))
                    k += 1
                elif generators_z[i, j]:
                    self.syndrome_circuit.append("CZ",
                                                 (0, 0, 1, i, 0, k),
                                                 (0, 0, 0, j))
                    k += 1
            self.syndrome_circuit.cur_time += 1

        w = len(self.syndrome_circuit.timepoints)-1
        for i in range(self.num_generators):
            self.syndrome_circuit.cur_time = w
            ng = self.syndrome_circuit.register[0, 0, 1, i, 0].num_qubits
            for j in range(ng-1-1, -1, -1):
                self.syndrome_circuit.append(
                    'CX', (0, 0, 1, i, 0, j), (0, 0, 1, i, 0, j+1))
            self.syndrome_circuit.append('H',  (0, 0, 1, i, 0, 0))
            self.syndrome_circuit.append('MR', (0, 0, 1, i, 0, 0))

        return self.syndrome_circuit
