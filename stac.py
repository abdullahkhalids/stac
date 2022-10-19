import numpy as np
from itertools import combinations
from IPython.display import display, Math


def print_matrix(array, augmented=False):
    '''
    Display an array using latex.
    If augmented=True, then a line is placed
    in the center of the matrix, which is useful
    for printing the stabilizer generator matrix.
    '''
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
        display(Math(f'\\left(\\begin{matname}{colalign}\n {data}\\end{matname}\\right)'))
    else:
        matname = '{pmatrix}'
        display(Math(f'\\begin{matname}\n {data}\\end{matname}'))


def _rref(A, colswap=True):
    '''
    Determine the set of elementary operations
    that give the reduced row echelon form (RREF) of A
    Returns the matrix rank, reduced matrix,
    and operations.
    '''
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
    '''
    Apply a set of elementary row operations, ops,
    to matrix A. If start_row is specified, then
    operations are performed relative to it.
    '''

    M = np.copy(A)

    for op in ops:
        if op[0] == "colswap":
            M[:, [op[1], op[2]]] = M[:, [op[2], op[1]]]
        elif op[0] == "rowswap":
            M[[start_row + op[1], start_row + op[2]], :] = M[[start_row + op[2], start_row + op[1]], :]
        elif op[0] == "addrow":
            M[start_row + op[1], :] = (M[start_row + op[1], :]
                                       + M[start_row + op[2], :]) % 2

    return M


def inner_product(v, w):

    n = int(len(v)/2)
    return (v[:n]@w[n:] + v[n:]@w[:n]) % 2


def print_pauli(G):
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
        print(pauli_str)


def circuit_to_qasm(circuit, custom_gates=''):
    qasm_str = ''

    qubits = 0

    for op in circuit:

        if op[1] > qubits:
            qubits = op[1]

        # op[0] is the gate name
        if op[0][:7].upper() == 'X_ERROR':
            continue
        elif op[0].upper() == 'R':
            op_str = 'reset q[{}];\n'.format(op[1])
        elif op[0].upper() == 'MR' or op[0].upper() == 'M':
            op_str = 'measure q[{}] -> c[{}];\n'.format(op[1], op[1])
        elif op[0].upper() == 'TICK':
            op_str = 'barrier '
            for q in op[1:]:
                op_str += 'q[{}],'.format(q)
            op_str = op_str[:-1] + ';\n'
        else:
            op_str = op[0].lower() + ' '
            # followed by one or two arguments
            if len(op) == 2:
                op_str += 'q[{}];\n'.format(op[1])
            else:
                op_str += 'q[{0}],q[{1}];\n'.format(op[1], op[2])
                if op[2] > qubits:
                    qubits = op[2]

        qasm_str += op_str

    qubits += 1
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + \
        custom_gates + \
        '\nqreg q[{}];\ncreg c[{}];\n'.format(qubits, qubits) + \
        qasm_str

    return qasm_str


def circuit_to_stim(circuit):
    stim_str = ''

    for op in circuit:
        print

        # op[0] is the gate name
        op_str = op[0].upper() + ' '

        if op[0].upper() == "TICK":
            op_str += '\n'
        elif len(op) == 2:
            op_str += '{} \n'.format(op[1])
        else:
            op_str += '{} {}\n'.format(op[1], op[2])

        stim_str += op_str

    return stim_str


def circuit_to_quirk(circuit):

    n = 0
    for op in circuit:
        if op[1] > n:
            n = op[1]

        if len(op) == 3 and op[2] > n:
            n = op[2]

    n += 1

    import json

    validops = set(['h', 'x', 'y', 'z', 'cx', 'cy', 'cz'])
    cols = []
    for op in circuit:
        if op[0] in validops:
            L = [1 for i in range(n)]
            if len(op) == 2:
                L[op[1]] = op[0].upper()
            else:
                L[op[1]] = "•"
                L[op[2]] = op[0][1].upper()

            cols.append(L)

    url = 'https://algassert.com/quirk#circuit={"cols":' + \
        json.dumps(cols, ensure_ascii=False) + '}'

    print(url)


def draw_circuit(circuit, custom_gates=None, **kwargs):
    from qiskit import QuantumCircuit
    from IPython.display import display
    if custom_gates is None:
        display(QuantumCircuit.from_qasm_str(
            circuit_to_qasm(circuit)).draw(**kwargs))
    else:
        display(QuantumCircuit.from_qasm_str(
            circuit_to_qasm(circuit, custom_gates)).draw(**kwargs))


def simulate_circuit(circuit,
                     n,
                     head=None,
                     incremental=False,
                     return_state=False,
                     print_state=True):

    from qiskit import QuantumCircuit, execute, Aer

    if head is None:
        head = ['basis', 'amplitude']

    tab = [[bin(i)[2:][-1::-1].ljust(n, '0')] for i in range(2**n)]

    cur_circ = []

    for ind in range(len(circuit)):
        op = circuit[ind]
        if (op[0].upper() == 'TICK' and incremental) or ind == len(circuit)-1:
            cur_circ.append(op)
            cur_circ.append(["id", n-1])
            qc = QuantumCircuit.from_qasm_str(circuit_to_qasm(cur_circ))
            job = execute(qc, Aer.get_backend('statevector_simulator'),
                          shots=1,
                          optimization_level=0)
            sv = job.result().get_statevector()
            amps = np.round(sv.data, 3)
            for i in range(2**n):
                tab[i].append(amps[i])

        else:
            cur_circ.append(op)

    if print_tab:
        display_states(head, tab)

    if return_tab:
        return tab


def display_states(head, *vars):
    import tabulate
    tabulate.PRESERVE_WHITESPACE = True
    import copy

    if len(vars) == 0:
        return
    else:
        comb_tab = copy.deepcopy(vars[0])
        for i in range(len(vars[0])):
            for v in vars[1:]:
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


class Code:

    def __init__(self, gens_x, gens_z):

        if gens_x.shape != gens_z.shape:
            print("The shape of the matrices don't match")
            return

        self.gens_x = gens_x
        self.gens_z = gens_z

        self.gens_mat = np.concatenate((self.gens_x, self.gens_z), axis=1)

        self.num_generators, self.num_physical_qubits = gens_x.shape
        self.num_logical_qubits = self.num_physical_qubits - self.num_generators
        self.distance = None

        self.rankx = None
        
        self.standard_gens_x = None
        self.standard_gens_z = None
        self.standard_gens_mat = None

        self.destab_gen_mat = None

        self.logical_xs = None
        self.logical_zs = None

        self.encoding_circuit = None
        self.decoding_circuit = None

        self.generators_qasm = None

    # def __repr__(self):
    #     pass
    #     # return 'Code({},{})'.format(self.gens_x, self_gens_z)

    def __str__(self):
        return 'A [[{},{}]] code'.format(self.num_physical_qubits,
                                         self.num_logical_qubits)

    def check_valid_code(self):
        is_valid = True
        for i in range(self.num_generators-1):
            for j in range(i+1, self.num_generators):
                if (self.gens_x[i]@self.gens_z[j] +
                    self.gens_x[j]@self.gens_z[i]) % 2:

                    print("Generators {} and {} don't commute".format(i, j))
                    print(np.append(self.gens_x[i], self.gens_z[i]))
                    print(np.append(self.gens_x[j], self.gens_z[j]))
                    is_valid = False

        return is_valid

    def standard_form(self):
        '''
        Take two matrices that describe the X and Z part
        of the stabilizer generator matrix.
        Transform the generator matrix to standard form.
        '''

        # Find RREF of X stabs
        (standard_gens_x, self.rankx, opsA) = _rref(self.gens_x)

        standard_gens_z = _perform_row_operations(self.gens_z, opsA)

        # now extract E' and reduce
        (rEp, rankEp, opsEp) = _rref(standard_gens_z[self.rankx:, self.rankx:])

        # perform same operations on full matrices
        self.standard_gens_x = _perform_row_operations(standard_gens_x,
                                                       opsEp,
                                                       self.rankx)
        self.standard_gens_z = _perform_row_operations(standard_gens_z,
                                                       opsEp,
                                                       self.rankx)

        self.standard_gens_mat = np.concatenate((self.standard_gens_x,
                                                 self.standard_gens_z),
                                                axis=1)

        return self.standard_gens_x, self.standard_gens_z, self.rankx

    def construct_logical_operators(self):
        '''Construct a set of logical operators for the code,
        using the Gottesman method.'''

        if self.standard_gens_x is None:
            self.standard_form()

        n = self.num_physical_qubits
        k = self.num_logical_qubits
        r = self.rankx

        # The relevant parts of the reduced generator matrix are
        A2 = self.standard_gens_x[0:r, (n-k):n]
        C1 = self.standard_gens_z[0:r, r:(n-k)]
        C2 = self.standard_gens_z[0:r, (n-k):n]
        E = self.standard_gens_z[r:(n-k), (n-k):n]

        # Construct the logical X operators
        self.logical_xs = np.concatenate((
            np.zeros((k, r), dtype=int),
            E.transpose(),
            np.identity(k, dtype=int),
            (E.transpose()@C1.transpose() + C2.transpose()) % 2,
            np.zeros((k, n-r), dtype=int)
        ), axis=1)

        # Construct the logical Z operators
        self.logical_zs = np.concatenate((
            np.zeros((k, n), dtype=int),
            A2.transpose(),
            np.zeros((k, n-k-r), dtype=int),
            np.identity(k, dtype=int)
        ), axis=1)

        return self.logical_xs, self.logical_zs

    def find_destabilizers(self):
        '''
        Find the destabilizers of standard form generators by exhaustive search.
        This will be slow for large codes but has the advantage that it will
        find the lowest weight destabilizers.
        '''

        if self.standard_gens_x is None:
            self.standard_form()
            
        n = self.num_physical_qubits

        destabs = np.empty((self.num_generators,2*n), dtype=int)
        destab_found = [False for i in range(self.num_generators)]
        i = 0
        b = False

        # these for loops create binary vector arrays
        # in increasing number of 1s.
        for k in range(1,2*n):
            if b:
                break
            for comb in combinations(np.arange(2*n), k):
                v = np.array([1 if i in comb else 0 for i in range(2*n)])

                # v should anti-commute with only one generator
                # to be a destabilizer
                num_anti_commute = 0
                for i in range(self.num_generators):
                    ip = inner_product(self.standard_gens_mat[i],v)
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

    def construct_encoding_circuit(self, fixed=False):
        '''Construct an encoding circuit for the code
        using Gottesman's method'''

        if self.logical_xs is None:
            self.construct_logical_operators()

        if self.destab_gen_mat is None:
            self.find_destabilizers()

        n = self.num_physical_qubits
        k = self.num_logical_qubits
        r = self.rankx

        self.encoding_circuit = []
        for i in range(k):
            for j in range(r, n-k):
                if self.logical_xs[i, j]:
                    self.encoding_circuit.append(["CX", n-k+i, j])

        for i in range(r):
            self.encoding_circuit.append(["H", i])
            for j in range(n):
                if i == j:
                    continue
                if self.standard_gens_x[i, j] and self.standard_gens_z[i, j]:
                    self.encoding_circuit.append(["CX", i, j])
                    self.encoding_circuit.append(["CZ", i, j])
                elif self.standard_gens_x[i, j]:
                    self.encoding_circuit.append(["CX", i, j])
                elif self.standard_gens_z[i, j]:
                    self.encoding_circuit.append(["CZ", i, j])

        if fixed:
            for i in range(3):
                for j in range(n):
                    if self.destab_gen_mat[i, j] and self.destab_gen_mat[i, n+j]:
                        self.encoding_circuit.append(["X", j])
                    elif self.destab_gen_mat[i, j]:
                        self.encoding_circuit.append(["X", j])
                    elif self.destab_gen_mat[i, n+j]:
                        self.encoding_circuit.append(["Z", j])

        return self.encoding_circuit

    def construct_decoding_circuit(self):
        '''Construct a decoding circuit for the code
        using Gottesman's method'''

        if self.logical_xs is None:
            self.construct_logical_operators()

        n = self.num_physical_qubits

        self.decoding_circuit = []

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
        if len(args) == 0:
            self.syndrome_circuit = self._construct_syndrome_circuit_simple(self.gens_x, self.gens_z)
        elif type(args[0]) is str:
            if args[0] == 'non_ft':
                self.syndrome_circuit = self._construct_syndrome_circuit_simple(self.gens_x, self.gens_z)
            elif args[0] == 'non_ft_standard':
                if self.standard_gens_x is None:
                    self.standard_form()
                self.syndrome_circuit = self._construct_syndrome_circuit_simple(self.standard_gens_x, self.standard_gens_z)

        return self.syndrome_circuit

    def _construct_syndrome_circuit_simple(self, gens_x, gens_z):
        n = self.num_physical_qubits
        # ancilla are from n n+m-1
        self.syndrome_circuit = []
        for i in range(self.num_generators):
            # first apply hadamard to ancilla
            self.syndrome_circuit.append(["H", n+i])

        self.syndrome_circuit.append(
            ["TICK"] +
            [i for i in range(self.num_physical_qubits,
                              self.num_physical_qubits +
                              self.num_generators)])

        for i in range(self.num_generators):
            for j in range(self.num_physical_qubits):
                if gens_x[i, j] and gens_z[i, j]:
                    self.syndrome_circuit.append(["CX", n+i, j])
                    self.syndrome_circuit.append(["CZ", n+i, j])
                elif gens_x[i, j]:
                    self.syndrome_circuit.append(["CX", n+i, j])
                elif gens_z[i, j]:
                    self.syndrome_circuit.append(["CZ", n+i, j])

            self.syndrome_circuit.append(
                ["TICK"] +
                [i for i in range(self.num_physical_qubits,
                                  self.num_physical_qubits +
                                  self.num_generators)])

        for i in range(self.num_generators):
            # last apply hadamard to ancilla
            self.syndrome_circuit.append(["H", n+i])

        for i in range(self.num_generators):
            self.syndrome_circuit.append(['MR', self.num_physical_qubits+i])

        return self.syndrome_circuit



    def generators_to_qasm(self):

        if self.standard_gens_x is None:
            self.standard_form()

        n = self.num_physical_qubits
        m = self.num_generators

        self.generators_qasm = []
        for i in range(m):
            circuit = []
            for j in range(n):
                if self.standard_gens_x[i, j] and self.standard_gens_z[i, j]:
                    circuit.append(["y", j])
                elif self.standard_gens_x[i, j]:
                    circuit.append(["x", j])
                elif self.standard_gens_z[i, j]:
                    circuit.append(["z", j])

            self.generators_qasm.append(circuit_to_qasm(circuit))

        return self.generators_qasm




class CommonCodes:


    def __init__(self):
        pass

    @classmethod
    def generate_code(cls, codename):
        if codename == '[[7,1,3]]':
            return cls._Steane()
        elif codename == '[[5,1,3]]':
            return cls._Code513()
        elif codename == '[[4,2,2]]':
            return cls._Code422()
        elif codename == '[[8,3,3]]':
            return cls._Code833()
        elif codename == '[[6,4,2]]':
            return cls._Code642()
        else:
            print("Code not found")

    @classmethod
    def _Steane(cls):
        hamming = np.array([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1]
        ], dtype=int)

        zeroM = np.zeros(hamming.shape, dtype=int)

        Sx = np.concatenate((hamming,zeroM))
        Sz = np.concatenate((zeroM,hamming))

        c = Code(Sx,Sz)
        c.distance = 3

        return c


    @classmethod
    def _Code513(cls):

        Sx = np.array([
            [1,0,0,1,0],
            [0,1,0,0,1],
            [1,0,1,0,0],
            [0,1,0,1,0]
        ], dtype=int)
        
        Sz = np.array([
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,1],
            [1,0,0,0,1]
        ], dtype=int)

        c = Code(Sx,Sz)
        c.distance = 3

        return c


    @classmethod
    def _Code833(cls):
        Sx = np.array([
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,1,0,1,1,0,1,0],
            [0,1,0,1,0,1,0,1],
            [0,1,1,0,1,0,0,1],
        ], dtype=int)
        Sz = np.array([
            [0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [0,0,1,1,0,0,1,1],
            [0,1,0,1,0,1,0,1],
        ], dtype=int)

        c = Code(Sx,Sz)
        c.distance = 3

        return c


    @classmethod
    def _Code422(cls):

        Sx = np.array([
            [1,0,0,1],
            [1,1,1,1]
        ], dtype=int)
        
        Sz = np.array([
            [0,1,1,0],
            [1,0,0,1]
        ], dtype=int)

        c = Code(Sx,Sz)
        c.distance = 2

        return c


    @classmethod
    def _Code642(cls):

        Sx = np.array([
            [1,1,1,1,1,1],
            [0,0,0,0,0,0]
        ], dtype=int)
        
        Sz = np.array([
            [0,0,0,0,0,0],
            [1,1,1,1,1,1]
        ], dtype=int)

        c = Code(Sx,Sz)
        c.distance = 2

        return c






