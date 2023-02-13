"""Provides a set of classes to define registers of qubits."""
from typing import Union, Optional, Iterator, Any, Sequence
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
    However, registers can be given any type.
    """

    def __init__(self) -> None:
        """
        Construct a Register.

        This class is generally not used directly, but its subclasses are.
        """
        self.index: Union[int, None] = None
        self.register_type: Union[str, None] = None

        self.elements: Sequence[Union['Register', VirtualQubit]] = []

        self.level: Union[int, None] = None

    def copy(self) -> 'Register':
        """
        Create a copy of this register.

        Returns
        -------
        Register
            The copy of this register.

        """
        reg = Register.__new__(Register)
        reg.index = self.index
        reg.register_type = self.register_type
        reg.elements = [r.copy() for r in self.elements]
        reg.level = self.level

        return reg

    def __repr__(self) -> str:
        """Return a representation of the object."""
        if type(self) is QubitRegister:
            t = 'QubitRegister'
        else:
            t = 'RegisterRegister'
        return ''.join([t,
                        f'(register_type={self.register_type}, ',
                        f'level={self.level}, ',
                        f'len={self.__len__()})'])

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.__repr__()

    def __len__(self) -> int:
        """Return number of objects in the register."""
        return len(self.elements)

    def __iter__(self) -> Iterator:
        """Return iterator of the Timepoint."""
        return self.elements.__iter__()

    def __getitem__(self,
                    s: Union[int, tuple]
                    ) -> Union['Register', VirtualQubit]:
        """
        Make Register subscriptable.

        Parameters
        ----------
        s : Union[int, tuple]
            Address of register to return.

        Returns
        -------
        Register
            The Register at s.

        Raises
        ------
        IndexError
            If s is not a valid address.
        """
        if type(s) is int:
            return self.elements.__getitem__(s)
        elif (type(s) is tuple
              and all(isinstance(v, int) for v in s[:-1])):  # type: ignore
            reg: Any = self
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

    def __ge__(self,
               other: 'Register') -> bool:
        """
        Determine if this Register contains qubits at every address as other.

        Parameters
        ----------
        other : Register
            The Register to compare self to.

        Returns
        -------
        bool
            True if this Register contains qubits as every address as other,
            otherwise False.

        This is an important check when adding circuits.
        """
        self_addresses = set(self.qubit_addresses())
        other_addresses = set(other.qubit_addresses())

        if other_addresses.issubset(self_addresses):
            return True
        else:
            return False

    def append(self,
               *registers: Union['Register', list['Register']]
               ) -> None:
        """
        Append one or more registers to this register.

        Parameters
        ----------
        *registers : Register
            Either a list of Registers, or pass one or more Registers as
            arguments.

        Raises
        ------
        TypeError
            If args are not a Register.

        """
        if (len(registers) == 1
                and issubclass(type(registers[0]), Register)):
            registers_list = [registers[0]]
        elif (len(registers) == 1 and type(registers[0]) is list
              and all(issubclass(type(r), Register) for r in registers[0])):
            registers_list = registers[0]  # type: ignore
        elif all(issubclass(type(r), Register) for r in registers):
            registers_list = list(registers)
        else:
            raise TypeError("Args must be registers or a list of registers.")

        for register in registers_list:
            register.index = len(self)  # type: ignore
            self.elements.append(register)  # type: ignore

    @ property
    def num_qubits(self) -> int:
        """
        Determine number of qubits in this Register recursively.

        Returns
        -------
        int
            Number of qubits.

        """
        if type(self) is QubitRegister:
            return len(self.elements)
        else:
            return sum([register.num_qubits  # type: ignore
                        for register in self.elements])

    def structure(self,
                  depth: int = -1) -> None:
        """
        Print the register structure.

        Parameters
        ----------
        max_depth : int, optional
            Determine structure to this depth. The default is -1, which goes to
            max depth.

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

    def check_address(self,
                      address: tuple) -> bool:
        """
        Determine if address is a valid qubit address in this Register.

        Parameters
        ----------
        address : tuple
            Address to be checked.

        Raises
        ------
        Exception
            If address not found, or if address does not point to a Qubit.

        Returns
        -------
        bool
            Only returns True if valid address, else raises Exception.

        """
        truncated_address: tuple = address[:-1]  # type: ignore
        try:
            self[truncated_address]
        except KeyError:
            raise Exception('Address not found.')

        if type(self[truncated_address]) is not QubitRegister:
            raise Exception(f'{self[truncated_address]} is not a qubit \
                            register.')

        try:
            self[address]
        except KeyError:
            raise Exception('Address not found.')

        return True

    def qubit_addresses(self,
                        my_address: Optional[tuple] = tuple()
                        ) -> list[tuple]:
        """
        Determine all qubit addresses within this Register, or its subregister.

        Parameters
        ----------
        my_address : tuple, optional
            The address of the subregister within which to search. The default
            is tuple(), which searches from the root of this Register.

        Returns
        -------
        address_list : list[tuple]
            List of addresses of qubits.

        """
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

    def qubits(self,
               register_type: Optional[str] = None
               ) -> Iterator:
        """
        Create generator for qubits within this Register.

        Parameters
        ----------
        register_type : str, optional
            Only generates qubits who or their parent register have this type.
            The default is None, in which case all qubits are generated.

        Yields
        ------
        Iterator
            Generator for qubits.

        """
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
                 register_type: str,
                 level: int,
                 num_qubits: int,
                 index: Optional[int] = None) -> None:
        """
        Construct a register to store qubits.

        Parameters
        ----------
        register_type : str
            The type of the Register. See Register class documenation.
        level : int
            The level within a Circuit at which this Register is present.
        num_qubits : int
            Number of qubits to create within this Register.
        index : int, optional
            The index of the Register within its parent. The default is None.

        """
        self.register_type = register_type
        self.level = level
        qubit_list = []
        for i in range(num_qubits):
            q = VirtualQubit(self.level,
                             i)
            qubit_list.append(q)

        self.elements = qubit_list

        self.index = index

    def copy(self) -> 'QubitRegister':
        """
        Create a copy of this register.

        Returns
        -------
        QubitRegister
            A copy of this register.

        """
        reg = QubitRegister.__new__(QubitRegister)
        reg.register_type = self.register_type
        reg.level = self.level
        reg.elements = [q.copy() for q in self.elements]
        reg.index = self.index

        return reg

    @ property  # type: ignore
    def index(self) -> Union[int, None]:
        """
        Get index of this Register.

        Returns
        -------
        int
            Index of this Register. Is None if not set.

        """
        return self._index

    @ index.setter
    def index(self,
              value: int) -> None:
        """
        Set index of this Register.

        Also sets the assigned_register property of each qubit within this
        Register.

        Parameters
        ----------
        value : int
            Value to set.

        """
        self._index = value
        # for qubit in self.elements:
        #     qubit.assigned_register = self._index


class RegisterRegister(Register):
    """Class to manipulate registers made out of subregisters."""

    def __init__(self,
                 register_type: str,
                 level: int,
                 subregisters: Optional[Iterator[Union[Register,
                                                       VirtualQubit]]] = None,
                 code: Optional[Any] = None) -> None:
        """
        Construct a register to hold registers.

        Parameters
        ----------
        register_type : str
            The type of the register. See Register class documenation.
        level : int
            The level within a Circuit at which this Register is present.
        subregisters : Iterator, optional
            If provided, these are appended to this Register. The default is
            None.
        code : Code, optional
            The code to attach to this Register. The default is None.

        """
        self.register_type = register_type
        self.level = level
        self.elements = []

        if type(subregisters) is list or type(subregisters) is tuple:
            self.elements = list(subregisters)  # type: ignore
        elif (type(subregisters) is QubitRegister
                or type(subregisters) is RegisterRegister):
            self.elements = list([subregisters])  # type: ignore
        for i, register in enumerate(self.elements):
            register.index = i

        self.code = code

        self._index: Union[int, None] = None

    def copy(self) -> 'RegisterRegister':
        """
        Create a copy of this register.

        Returns
        -------
        RegisterRegister
            The copied register.

        """
        reg = RegisterRegister.__new__(RegisterRegister)

        reg.register_type = self.register_type
        reg.level = self.level
        reg.elements = [r.copy() for r in self.elements]
        reg.code = self.code
        reg._index = self._index

        return reg

    @ property  # type: ignore
    def index(self) -> Optional[int]:
        """
        Get index of this Register.

        Returns
        -------
        int
            Index of this Register. Is None if not set.

        """
        return self._index

    @ index.setter
    def index(self,
              value: int) -> None:
        """
        Set index of this Register.

        Parameters
        ----------
        value : int
            Value to set.

        """
        self._index = value

    def constituent_register_mapping(self) -> list[list]:
        """
        Determine the constituent register of every qubit within this Register.

        Returns
        -------
        list[list]
            List of lists, where each sublist is a
            [address, constituent_reigster] pair.

        """
        mapping = []

        for address in reversed(self.qubit_addresses()):
            if self[address].constituent_register is not None:  # type: ignore
                mapping.append([
                    address,
                    self[address].constituent_register])  # type: ignore

        return mapping
