"""Provides a set of classes to define registers of qubits."""
from .qubit import VirtualQubit


class Register:
    """
    Class to create and manipulate registers.

    Registers have a type that determines how the register is functionally
    used within the fault-tolerant circuit. Stac recognizes the following
    types:
        d : Data registers store encoded qubits. For a [[n,k,d]] code, the
            size of such registers should be n.
        g : Stabilizer generator measurement registers have the ancilla qubits
            used to measure one stabilizer generator of a code. The size of
            such registers is usually equal to the weight of the generator.
        s : Syndrome measurement registers are a collection of g-type
            registers.
        e : Encoded qubit registers usually contain one d-type register and
            one s-type register.
    However, these types are not enforced in any way, and the registers can be
    given any type.
    """

    def __init__(self):

        self.index = None
        self.register_type = None

        self.elements = []

        self.level = None

    def __repr__(self):
        if type(self) is QubitRegister:
            t = 'QubitRegister'
        else:
            t = 'RegisterRegister'
        return ''.join([t,
                        f'(register_type={self.register_type}, ',
                        f'level={self.level}, ',
                        f'len={self.__len__()})'])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return self.elements.__iter__()

    def __getitem__(self, s):
        """Make register subscriptable."""
        if type(s) is int:
            return self.elements.__getitem__(s)
        elif type(s) is tuple and all(map(lambda v: type(v) is int, s[:-1])):
            reg = self
            for t in s:
                try:
                    reg = reg.elements[t]
                except IndexError:
                    error_message = f'The register does not contain a \
subregister or qubit at {s}.'
                    raise IndexError(error_message)
            return reg
        else:
            raise TypeError('Cannot recognize subscript.')

    def __ge__(self, other):

        self_addresses = set(self.qubit_addresses())
        other_addresses = set(other.qubit_addresses())

        if other_addresses.issubset(self_addresses):
            return True
        else:
            return False

    def append(self, *registers):
        # apply appropriate checks
        if len(registers) == 1:
            if (type(registers[0]) is RegisterRegister
                    or type(registers[0]) is QubitRegister):
                registers_list = [registers[0]]
        else:
            registers_list = registers

        for register in registers_list:
            register.index = len(self)
            self.elements.append(register)

    @property
    def num_qubits(self):
        if type(self) is QubitRegister:
            return len(self.elements)
        else:
            return sum([register.num_qubits for register in self.elements])

    def structure(self, depth=-1):
        """
        Print the register structure.

        Parameters
        ----------
        max_depth : TYPE, optional
            DESCRIPTION. The default is -1.

        """
        if depth == -1:
            depth = 4294967295

        def determine_structure(register, indent, d):
            if d > depth:
                return ''
            s = ' '*indent
            s += ' '.join([str(register.index),
                           register.register_type,
                           'x',
                           str(len(register)),
                           '\n'])
            if type(register) is not QubitRegister:
                for child_register in register.elements:
                    s += determine_structure(child_register, indent+3, d+1)
            return s
        print(determine_structure(self, 0, 0))

    def check_address(self, address):
        truncated_address = address[:-1]
        try:
            self[truncated_address]
        except KeyError:
            raise Exception('Address not found')

        if type(self[truncated_address]) is not QubitRegister:
            raise Exception('Not a qubit register')

        try:
            self[address]
        except KeyError:
            raise Exception('Address not found')

        return True

    def qubit_addresses(self, my_address=tuple()):
        address_list = []

        def determine_structure(register, my_address):
            if type(register) is RegisterRegister:
                for i, child_register in enumerate(register.elements):
                    determine_structure(child_register, my_address + (i,))
            else:
                for i, qubit in enumerate(register.elements):
                    address_list.append(my_address + (i,))
        determine_structure(self, tuple())
        return address_list

    def qubits(self, register_type=None):
        def iterator(register, certain_yield=False):
            if type(register) is RegisterRegister:
                for child_register in register:
                    if (child_register.register_type == register_type
                            or certain_yield):
                        yield from iterator(child_register, True)
                    else:
                        yield from iterator(child_register)
            elif (register_type is None
                  or certain_yield
                  or register.register_type == register_type):
                for qubit in register:
                    yield qubit
        if self.register_type == register_type:
            return iterator(self, True)
        else:
            return iterator(self)


class QubitRegister(Register):
    """Class to manipulate registers made out of virtual qubits."""

    def __init__(self,
                 register_type,
                 level,
                 num_qubits,
                 index=None):

        self.register_type = register_type
        self.level = level
        qubit_list = []
        for i in range(num_qubits):
            q = VirtualQubit(self.level,
                             i)
            qubit_list.append(q)

        self.elements = qubit_list

        self.index = index

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        for qubit in self.elements:
            qubit.assigned_register = self._index


class RegisterRegister(Register):
    """Class to manipulate registers made out of subregisters."""

    def __init__(self,
                 register_type,
                 level,
                 subregisters=None,
                 code=None):

        self.register_type = register_type
        self.level = level
        self.elements = []

        if type(subregisters) is list or type(subregisters) is tuple:
            self.elements = list(subregisters)
        elif (type(subregisters) is QubitRegister
                or type(subregisters) is RegisterRegister):
            self.elements = list([subregisters])
        for i, register in enumerate(self.elements):
            register.index = i

        self.code = code

        self.index = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    # def _create_encoded_qubit_register(self):
    #     n = self.attached_code.num_physical_qubits
    #     datareg = QubitRegister('d', self.level, n)
    #     genregs = [QubitRegister('g', self.level, sum(g))
    #                for g in self.attached_code.generator_matrix]
    #     syndreg = RegisterRegister('s', 0, genregs)
    #     self.append(datareg, syndreg)

    def constituent_register_mapping(self):

        mapping = []

        for address in reversed(self.qubit_addresses()):
            if self[address].constituent_register is not None:
                mapping.append([address, self[address].constituent_register])

        return mapping
