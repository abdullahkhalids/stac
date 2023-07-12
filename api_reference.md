# Api Reference

* [display\_states](#stac.circuit.display_states)
* [Circuit](#stac.circuit.Circuit)
  * [\_\_init\_\_](#stac.circuit.Circuit.__init__)
  * [simple](#stac.circuit.Circuit.simple)
  * [\_\_repr\_\_](#stac.circuit.Circuit.__repr__)
  * [\_\_str\_\_](#stac.circuit.Circuit.__str__)
  * [\_\_len\_\_](#stac.circuit.Circuit.__len__)
  * [\_\_iter\_\_](#stac.circuit.Circuit.__iter__)
  * [\_\_getitem\_\_](#stac.circuit.Circuit.__getitem__)
  * [reverse](#stac.circuit.Circuit.reverse)
  * [clear](#stac.circuit.Circuit.clear)
  * [cur\_time](#stac.circuit.Circuit.cur_time)
  * [cur\_time](#stac.circuit.Circuit.cur_time)
  * [append](#stac.circuit.Circuit.append)
  * [geo\_append](#stac.circuit.Circuit.geo_append)
  * [append\_register](#stac.circuit.Circuit.append_register)
  * [map\_to\_physical\_layout](#stac.circuit.Circuit.map_to_physical_layout)
  * [num\_qubits](#stac.circuit.Circuit.num_qubits)
  * [apply\_circuit](#stac.circuit.Circuit.apply_circuit)
  * [\_\_add\_\_](#stac.circuit.Circuit.__add__)
  * [\_\_mul\_\_](#stac.circuit.Circuit.__mul__)
  * [qasm](#stac.circuit.Circuit.qasm)
  * [stim](#stac.circuit.Circuit.stim)
  * [quirk](#stac.circuit.Circuit.quirk)
  * [simulate](#stac.circuit.Circuit.simulate)
  * [sample](#stac.circuit.Circuit.sample)
  * [draw](#stac.circuit.Circuit.draw)
* [Annotation](#stac.annotation.Annotation)
  * [\_\_init\_\_](#stac.annotation.Annotation.__init__)
  * [\_\_repr\_\_](#stac.annotation.Annotation.__repr__)
  * [\_\_str\_\_](#stac.annotation.Annotation.__str__)
  * [copy](#stac.annotation.Annotation.copy)
* [AnnotationSlice](#stac.annotation.AnnotationSlice)
  * [\_\_init\_\_](#stac.annotation.AnnotationSlice.__init__)
  * [\_\_repr\_\_](#stac.annotation.AnnotationSlice.__repr__)
  * [\_\_str\_\_](#stac.annotation.AnnotationSlice.__str__)
  * [\_\_iter\_\_](#stac.annotation.AnnotationSlice.__iter__)
  * [\_\_getitem\_\_](#stac.annotation.AnnotationSlice.__getitem__)
  * [\_\_len\_\_](#stac.annotation.AnnotationSlice.__len__)
  * [copy](#stac.annotation.AnnotationSlice.copy)
  * [append](#stac.annotation.AnnotationSlice.append)
  * [\_\_add\_\_](#stac.annotation.AnnotationSlice.__add__)
  * [\_\_iadd\_\_](#stac.annotation.AnnotationSlice.__iadd__)
* [hexagon\_coordinates](#stac.old_colorcode.hexagon_coordinates)
* [create\_hexagon\_svg](#stac.old_colorcode.create_hexagon_svg)
* [ColorCode](#stac.old_colorcode.ColorCode)
  * [\_\_init\_\_](#stac.old_colorcode.ColorCode.__init__)
  * [construct\_logical\_operators](#stac.old_colorcode.ColorCode.construct_logical_operators)
* [Operation](#stac.operation.Operation)
  * [\_\_init\_\_](#stac.operation.Operation.__init__)
  * [\_\_repr\_\_](#stac.operation.Operation.__repr__)
  * [\_\_str\_\_](#stac.operation.Operation.__str__)
  * [\_\_eq\_\_](#stac.operation.Operation.__eq__)
  * [\_\_hash\_\_](#stac.operation.Operation.__hash__)
  * [copy](#stac.operation.Operation.copy)
  * [rebase\_qubits](#stac.operation.Operation.rebase_qubits)
* [ColorCode](#stac.colorcode.ColorCode)
  * [\_\_init\_\_](#stac.colorcode.ColorCode.__init__)
  * [construct\_logical\_operators](#stac.colorcode.ColorCode.construct_logical_operators)
* [Instruction](#stac.instruction.Instruction)
* [Register](#stac.register.Register)
  * [\_\_init\_\_](#stac.register.Register.__init__)
  * [copy](#stac.register.Register.copy)
  * [\_\_repr\_\_](#stac.register.Register.__repr__)
  * [\_\_str\_\_](#stac.register.Register.__str__)
  * [\_\_len\_\_](#stac.register.Register.__len__)
  * [\_\_iter\_\_](#stac.register.Register.__iter__)
  * [\_\_getitem\_\_](#stac.register.Register.__getitem__)
  * [\_\_ge\_\_](#stac.register.Register.__ge__)
  * [append](#stac.register.Register.append)
  * [num\_qubits](#stac.register.Register.num_qubits)
  * [structure](#stac.register.Register.structure)
  * [check\_address](#stac.register.Register.check_address)
  * [qubit\_addresses](#stac.register.Register.qubit_addresses)
  * [qubits](#stac.register.Register.qubits)
* [QubitRegister](#stac.register.QubitRegister)
  * [\_\_init\_\_](#stac.register.QubitRegister.__init__)
  * [copy](#stac.register.QubitRegister.copy)
  * [index](#stac.register.QubitRegister.index)
  * [index](#stac.register.QubitRegister.index)
* [RegisterRegister](#stac.register.RegisterRegister)
  * [\_\_init\_\_](#stac.register.RegisterRegister.__init__)
  * [copy](#stac.register.RegisterRegister.copy)
  * [index](#stac.register.RegisterRegister.index)
  * [index](#stac.register.RegisterRegister.index)
  * [constituent\_register\_mapping](#stac.register.RegisterRegister.constituent_register_mapping)
* [ConcatCode](#stac.concatenation.ConcatCode)
  * [\_\_init\_\_](#stac.concatenation.ConcatCode.__init__)
* [print\_matrix](#stac.code.print_matrix)
* [print\_paulis](#stac.code.print_paulis)
* [print\_paulis\_indexed](#stac.code.print_paulis_indexed)
* [Code](#stac.code.Code)
  * [\_\_init\_\_](#stac.code.Code.__init__)
  * [\_\_repr\_\_](#stac.code.Code.__repr__)
  * [\_\_str\_\_](#stac.code.Code.__str__)
  * [check\_valid\_code](#stac.code.Code.check_valid_code)
  * [check\_in\_normalizer](#stac.code.Code.check_in_normalizer)
  * [construct\_standard\_form](#stac.code.Code.construct_standard_form)
  * [construct\_logical\_operators](#stac.code.Code.construct_logical_operators)
  * [construct\_logical\_gate\_circuits](#stac.code.Code.construct_logical_gate_circuits)
  * [find\_destabilizers](#stac.code.Code.find_destabilizers)
  * [construct\_data\_register](#stac.code.Code.construct_data_register)
  * [construct\_syndrome\_measurement\_register](#stac.code.Code.construct_syndrome_measurement_register)
  * [construct\_encoded\_qubit\_register](#stac.code.Code.construct_encoded_qubit_register)
  * [construct\_encoding\_circuit](#stac.code.Code.construct_encoding_circuit)
  * [construct\_decoding\_circuit](#stac.code.Code.construct_decoding_circuit)
  * [construct\_syndrome\_circuit](#stac.code.Code.construct_syndrome_circuit)
  * [construct\_encoded\_qubit](#stac.code.Code.construct_encoded_qubit)
* [InstructionBlock](#stac.instructionblock.InstructionBlock)
  * [\_\_repr\_\_](#stac.instructionblock.InstructionBlock.__repr__)
  * [\_\_str\_\_](#stac.instructionblock.InstructionBlock.__str__)
  * [\_\_iter\_\_](#stac.instructionblock.InstructionBlock.__iter__)
  * [\_\_getitem\_\_](#stac.instructionblock.InstructionBlock.__getitem__)
  * [\_\_len\_\_](#stac.instructionblock.InstructionBlock.__len__)
  * [insert](#stac.instructionblock.InstructionBlock.insert)
  * [copy](#stac.instructionblock.InstructionBlock.copy)
  * [append](#stac.instructionblock.InstructionBlock.append)
* [AnnotationBlock](#stac.instructionblock.AnnotationBlock)
* [RepetitionBlock](#stac.instructionblock.RepetitionBlock)
* [IfBlock](#stac.instructionblock.IfBlock)
* [PhysicalQubit](#stac.qubit.PhysicalQubit)
  * [\_\_init\_\_](#stac.qubit.PhysicalQubit.__init__)
* [VirtualQubit](#stac.qubit.VirtualQubit)
  * [\_\_init\_\_](#stac.qubit.VirtualQubit.__init__)
  * [index\_in\_assigned\_register](#stac.qubit.VirtualQubit.index_in_assigned_register)
  * [index\_in\_assigned\_register](#stac.qubit.VirtualQubit.index_in_assigned_register)
  * [index](#stac.qubit.VirtualQubit.index)
  * [index](#stac.qubit.VirtualQubit.index)
  * [copy](#stac.qubit.VirtualQubit.copy)
* [Timepoint](#stac.timepoint.Timepoint)
  * [\_\_init\_\_](#stac.timepoint.Timepoint.__init__)
  * [\_\_repr\_\_](#stac.timepoint.Timepoint.__repr__)
  * [\_\_str\_\_](#stac.timepoint.Timepoint.__str__)
  * [\_\_iter\_\_](#stac.timepoint.Timepoint.__iter__)
  * [\_\_getitem\_\_](#stac.timepoint.Timepoint.__getitem__)
  * [\_\_len\_\_](#stac.timepoint.Timepoint.__len__)
  * [copy](#stac.timepoint.Timepoint.copy)
  * [append](#stac.timepoint.Timepoint.append)
  * [can\_append](#stac.timepoint.Timepoint.can_append)
  * [rebase\_qubits](#stac.timepoint.Timepoint.rebase_qubits)
  * [can\_add](#stac.timepoint.Timepoint.can_add)
  * [\_\_add\_\_](#stac.timepoint.Timepoint.__add__)
  * [\_\_iadd\_\_](#stac.timepoint.Timepoint.__iadd__)
* [CommonCodes](#stac.commoncodes.CommonCodes)
  * [\_\_init\_\_](#stac.commoncodes.CommonCodes.__init__)
  * [generate\_code](#stac.commoncodes.CommonCodes.generate_code)

<a id="stac.circuit.display_states"></a>

#### display\_states

```python
def display_states(head, *args: list[list])
```

```
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
```

<a id="stac.circuit.Circuit"></a>

## Circuit

```python
class Circuit()
```

```
    Class for creating and manipulating quantum circuits.
```

<a id="stac.circuit.Circuit.__init__"></a>

#### Circuit.\_\_init\_\_

```python
def __init__(self, *args: Any) -> None
```

```
    Construct a quantum circuit.
    
    Parameters
    ----------
    Register:
        If passed, then the Register is appended to the circuit.
```

<a id="stac.circuit.Circuit.simple"></a>

#### Circuit.simple

```python
@staticmethod
def simple(num_qubits: int) -> 'Circuit'
```

```
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
```

<a id="stac.circuit.Circuit.__repr__"></a>

#### Circuit.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.circuit.Circuit.__str__"></a>

#### Circuit.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.circuit.Circuit.__len__"></a>

#### Circuit.\_\_len\_\_

```python
def __len__(self) -> int
```

```
    Return number of operations in the quantum circuit.
```

<a id="stac.circuit.Circuit.__iter__"></a>

#### Circuit.\_\_iter\_\_

```python
def __iter__(self) -> Iterator
```

```
    Return iterator for the quantum circuit.
```

<a id="stac.circuit.Circuit.__getitem__"></a>

#### Circuit.\_\_getitem\_\_

```python
def __getitem__(self, ind: int) -> Operation
```

```
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
```

<a id="stac.circuit.Circuit.reverse"></a>

#### Circuit.reverse

```python
def reverse(self) -> 'Circuit'
```

```
    Return a circuit in which all operations are reversed.
```

<a id="stac.circuit.Circuit.clear"></a>

#### Circuit.clear

```python
def clear(self) -> None
```

```
    Remove all operations and annotations from circuit.
```

<a id="stac.circuit.Circuit.cur_time"></a>

#### Circuit.cur\_time

```python
@property
def cur_time(self) -> int
```

```
    Return time at which new operations will begin getting added.
    
    Returns
    -------
    int
        Current time.
```

<a id="stac.circuit.Circuit.cur_time"></a>

#### Circuit.cur\_time

```python
@cur_time.setter
def cur_time(self, new_time: int) -> None
```

```
    Set the current time in the circuit.
    
    Parameters
    ----------
    new_time : int
        The time to set.
```

<a id="stac.circuit.Circuit.append"></a>

#### Circuit.append

```python
def append(self, *args, time=None)
```

```
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
```

<a id="stac.circuit.Circuit.geo_append"></a>

#### Circuit.geo\_append

```python
def geo_append(self, *args, time=None)
```

```
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
```

<a id="stac.circuit.Circuit.append_register"></a>

#### Circuit.append\_register

```python
def append_register(self, register: Register) -> tuple
```

```
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
```

<a id="stac.circuit.Circuit.map_to_physical_layout"></a>

#### Circuit.map\_to\_physical\_layout

```python
def map_to_physical_layout(self,
                           layout: Optional[str] = 'linear') -> list[list]
```

```
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
```

<a id="stac.circuit.Circuit.num_qubits"></a>

#### Circuit.num\_qubits

```python
@property
def num_qubits(self) -> int
```

```
    Determine number of qubits in circuit at level 0.
    
    Returns
    -------
    int
        Number of qubits.
```

<a id="stac.circuit.Circuit.apply_circuit"></a>

#### Circuit.apply\_circuit

```python
def apply_circuit(self,
                  other: 'Circuit',
                  new_base: tuple,
                  time: Optional[Union[int, list[int]]] = None) -> None
```

```
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
```

<a id="stac.circuit.Circuit.__add__"></a>

#### Circuit.\_\_add\_\_

```python
def __add__(self, other: 'Circuit') -> 'Circuit'
```

```
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
```

<a id="stac.circuit.Circuit.__mul__"></a>

#### Circuit.\_\_mul\_\_

```python
def __mul__(self, repetitions: int) -> 'Circuit'
```

```
    Create a circuit which repeates repetitions times.
    
    Parameters
    ----------
    repetitions : int
        The number of repetitions.
    
    Returns
    -------
    new_circuit : Circuit
        The repeated circuit.
```

<a id="stac.circuit.Circuit.qasm"></a>

#### Circuit.qasm

```python
def qasm(self) -> str
```

```
    Convert circuit to qasm string.
    
    Returns
    -------
    qasm_str : str
        The qasm string of the circuit.
```

<a id="stac.circuit.Circuit.stim"></a>

#### Circuit.stim

```python
def stim(self, clean: bool = False) -> str
```

```
    Convert circuit to a string that can be imported by stim.
    
    Parameters
    ----------
    clean : bool
        If True, then pass it through stim to compactify it.
    
    Returns
    -------
    stim_str : str
        A string suitable for importing by stim.
```

<a id="stac.circuit.Circuit.quirk"></a>

#### Circuit.quirk

```python
def quirk(self) -> None
```

```
    Convert circuit to a quirk circuit.
    
    Returns
    -------
    None.
    Prints a url that can opened in the browser.
```

<a id="stac.circuit.Circuit.simulate"></a>

#### Circuit.simulate

```python
def simulate(self,
             head: Optional[list[str]] = None,
             incremental: bool = False,
             return_state: bool = False,
             print_state: bool = True) -> list[Any]
```

```
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
```

<a id="stac.circuit.Circuit.sample"></a>

#### Circuit.sample

```python
def sample(self,
           samples=1,
           return_sample: bool = False,
           print_sample: bool = True) -> list[int]
```

```
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
```

<a id="stac.circuit.Circuit.draw"></a>

#### Circuit.draw

```python
def draw(self,
         medium: str = 'svg',
         filename: str = None,
         *,
         highlight_timepoints: Optional[bool] = False) -> None
```

```
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
```

<a id="stac.annotation.Annotation"></a>

## Annotation

```python
class Annotation(Instruction)
```

```
    Class to represent circuit annotations.
```

<a id="stac.annotation.Annotation.__init__"></a>

#### Annotation.\_\_init\_\_

```python
def __init__(self, name: str, targets: list = []) -> None
```

```
    Construct annotation object.
    
    Parameters
    ----------
    name : str
        Name of annotation.
    targets : list, optional
        Any targets this annotation has. The default is [].
```

<a id="stac.annotation.Annotation.__repr__"></a>

#### Annotation.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.annotation.Annotation.__str__"></a>

#### Annotation.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.annotation.Annotation.copy"></a>

#### Annotation.copy

```python
def copy(self) -> 'Annotation'
```

```
    Return copy of object.
```

<a id="stac.annotation.AnnotationSlice"></a>

## AnnotationSlice

```python
class AnnotationSlice()
```

```
    Class to create and manipulate annotation slices.
```

<a id="stac.annotation.AnnotationSlice.__init__"></a>

#### AnnotationSlice.\_\_init\_\_

```python
def __init__(self, new_ann: Annotation = None) -> None
```

```
    Construct an AnnotationSlice.
    
    Parameters
    ----------
    new_ann : Annotation, optional
        This annotation will be appended to this slice. The default is
        None.
```

<a id="stac.annotation.AnnotationSlice.__repr__"></a>

#### AnnotationSlice.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.annotation.AnnotationSlice.__str__"></a>

#### AnnotationSlice.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.annotation.AnnotationSlice.__iter__"></a>

#### AnnotationSlice.\_\_iter\_\_

```python
def __iter__(self) -> Iterator[Annotation]
```

```
    Return iterator of the AnnotationSlice.
```

<a id="stac.annotation.AnnotationSlice.__getitem__"></a>

#### AnnotationSlice.\_\_getitem\_\_

```python
def __getitem__(self, ind) -> Union[Annotation, list[Annotation]]
```

```
    Make Timepoint subscriptable.
```

<a id="stac.annotation.AnnotationSlice.__len__"></a>

#### AnnotationSlice.\_\_len\_\_

```python
def __len__(self) -> int
```

```
    Return number of annotations in the AnnotationSlice.
```

<a id="stac.annotation.AnnotationSlice.copy"></a>

#### AnnotationSlice.copy

```python
def copy(self) -> 'AnnotationSlice'
```

```
    Return a copy of the AnnotationSlice.
```

<a id="stac.annotation.AnnotationSlice.append"></a>

#### AnnotationSlice.append

```python
def append(self, new_ann: Annotation) -> None
```

```
    Append operation to this AnnotationSlice.
    
    Parameters
    ----------
    new_ann : Annotation
        Annotation to append.
```

<a id="stac.annotation.AnnotationSlice.__add__"></a>

#### AnnotationSlice.\_\_add\_\_

```python
def __add__(self, other: 'AnnotationSlice') -> 'AnnotationSlice'
```

```
    Create sum of this AnnotationSlice and other AnnotationSlice.
    
    Parameters
    ----------
    other : AnnotationSlice
        AnnotationSlice to be added.
    
    Returns
    -------
    anns : AnnotationSlice
        Summed AnnotationSlice.
```

<a id="stac.annotation.AnnotationSlice.__iadd__"></a>

#### AnnotationSlice.\_\_iadd\_\_

```python
def __iadd__(self, other: 'AnnotationSlice') -> 'AnnotationSlice'
```

```
    Add other AnnotationSlice to this AnnotationSlice.
    
    Parameters
    ----------
    other : AnnotationSlice
        AnnotationSlice to be added.
    
    Returns
    -------
    AnnotationSlice
        Summed AnnotationSlice.
```

<a id="stac.old_colorcode.hexagon_coordinates"></a>

#### hexagon\_coordinates

```python
def hexagon_coordinates(x0: float,
                        y0: float,
                        size: int = 25,
                        included: str = 'full') -> list[tuple[float, float]]
```

```
    Determine the coordinates of the vertices of a hexagon.
    
    The hexagon is oriented so there are horizontal sides on the top and
    bottom.
    
    Parameters
    ----------
    x0 : float
        Horizontal position of center.
    y0 : float
        Vertical position of center.
    size : int, optional
        The radius of a circuit touching the vertices of the hexagon. The
        default is 25.
    included : str, optional
        Which vertices to include in the output. The options are full, top,
        bottom, left and right. The default is 'full'.
    
    Returns
    -------
    list(tuple(float))
        A list of (x,y) coordinates. Go clockwise from the right-most vertex.
```

<a id="stac.old_colorcode.create_hexagon_svg"></a>

#### create\_hexagon\_svg

```python
def create_hexagon_svg(x0: float,
                       y0: float,
                       color: str,
                       size: int = 25,
                       included: str = 'full')
```

```
    Create an svg.Polygon object of a hexagon.
    
    Parameters
    ----------
    x0 : float
        Horizontal position of center.
    y0 : float
        Vertical position of center.
    color : str
        Options are r, g or b.
    size : int, optional
        The radius of a circuit touching the vertices of the hexagon. The
        default is 25.
    included : str, optional
        Which vertices to include in the output. The options are full, top,
        bottom, left and right. The default is 'full'.
    
    Returns
    -------
    pg : svg.Polygon
        The svg.Polygon object of the hexagon.
```

<a id="stac.old_colorcode.ColorCode"></a>

## ColorCode

```python
class ColorCode(Code)
```

```
    Class for creating color codes.
```

<a id="stac.old_colorcode.ColorCode.__init__"></a>

#### ColorCode.\_\_init\_\_

```python
def __init__(self,
             distance: int,
             geometry: str = "hexagonal",
             color_order: list[str] = ['g', 'r', 'b']) -> None
```

```
    Construct the color code of some geometry and distance.
    
    Parameters
    ----------
    distance : int
        The distance of the code.
    geometry : str, optional
        Describes the shape of the primal lattice. The default and only
        option currently is "hexagonal".
    color_order: str, optional
        Order of colors in the lattice.
```

<a id="stac.old_colorcode.ColorCode.construct_logical_operators"></a>

#### ColorCode.construct\_logical\_operators

```python
def construct_logical_operators(self,
                                method: str = "boundary: 2") -> (Any, Any)
```

```
    Constructs logical operators of the code.
    
    Parameters
    ----------
    method : str, optional
        With boundaries with color 0, 1, 2. The options are:
            "boundary: green"
            "boundary: red"
            "boundary: blue" (default)
            "gottesman" (generic method)
    
    Returns
    -------
    logical_xs: numpy.array
        Array of logical xs. Each row is an operator.
    logical_zs: numpy.array
        Array of logical xs. Each row is an operator.
```

<a id="stac.operation.Operation"></a>

## Operation

```python
class Operation(Instruction)
```

```
    Class to represent operations in circuits.
```

<a id="stac.operation.Operation.__init__"></a>

#### Operation.\_\_init\_\_

```python
def __init__(self,
             name: str,
             targets: list = [],
             parameters: list[float] | None = None) -> None
```

```
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
```

<a id="stac.operation.Operation.__repr__"></a>

#### Operation.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.operation.Operation.__str__"></a>

#### Operation.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.operation.Operation.__eq__"></a>

#### Operation.\_\_eq\_\_

```python
def __eq__(self, other: 'Operation') -> bool
```

```
    Determine if two operations are equal.
```

<a id="stac.operation.Operation.__hash__"></a>

#### Operation.\_\_hash\_\_

```python
def __hash__(self) -> int
```

```
    Return a hash of the object.
```

<a id="stac.operation.Operation.copy"></a>

#### Operation.copy

```python
def copy(self) -> 'Operation'
```

```
    Return copy of class.
```

<a id="stac.operation.Operation.rebase_qubits"></a>

#### Operation.rebase\_qubits

```python
def rebase_qubits(self, new_base: tuple) -> 'Operation'
```

```
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
```

<a id="stac.colorcode.ColorCode"></a>

## ColorCode

```python
class ColorCode(Code)
```

```
    Class for creating triangular color codes.
```

<a id="stac.colorcode.ColorCode.__init__"></a>

#### ColorCode.\_\_init\_\_

```python
def __init__(self,
             distance: int,
             geometry: str = "hexagonal",
             color_order: list[str] = ['g', 'r', 'b']) -> None
```

```
    Construct the color code of some geometry and distance.
    
    Parameters
    ----------
    distance : int
        The distance of the code.
    geometry : str, optional
        Describes the shape of the primal lattice. The default and only
        option currently is "hexagonal".
    color_order: str, optional
        Order of colors in the lattice.
```

<a id="stac.colorcode.ColorCode.construct_logical_operators"></a>

#### ColorCode.construct\_logical\_operators

```python
def construct_logical_operators(self,
                                method: str = "boundary: 2") -> (Any, Any)
```

```
    Constructs logical operators of the code.
    
    Parameters
    ----------
    method : str, optional
        With boundaries with color 0, 1, 2. The options are:
            "boundary: green"
            "boundary: red"
            "boundary: blue" (default)
            "gottesman" (generic method)
    
    Returns
    -------
    logical_xs: numpy.array
        Array of logical xs. Each row is an operator.
    logical_zs: numpy.array
        Array of logical xs. Each row is an operator.
```

<a id="stac.instruction.Instruction"></a>

## Instruction

```python
class Instruction()
```

```
    Class to represent all circuit operations, annotations etc.
```

<a id="stac.register.Register"></a>

## Register

```python
class Register()
```

```
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
```

<a id="stac.register.Register.__init__"></a>

#### Register.\_\_init\_\_

```python
def __init__(self) -> None
```

```
    Construct a Register.
    
    This class is generally not used directly, but its subclasses are.
```

<a id="stac.register.Register.copy"></a>

#### Register.copy

```python
def copy(self) -> 'Register'
```

```
    Create a copy of this register.
    
    Returns
    -------
    Register
        The copy of this register.
```

<a id="stac.register.Register.__repr__"></a>

#### Register.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.register.Register.__str__"></a>

#### Register.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.register.Register.__len__"></a>

#### Register.\_\_len\_\_

```python
def __len__(self) -> int
```

```
    Return number of objects in the register.
```

<a id="stac.register.Register.__iter__"></a>

#### Register.\_\_iter\_\_

```python
def __iter__(self) -> Iterator
```

```
    Return iterator of the Register.
```

<a id="stac.register.Register.__getitem__"></a>

#### Register.\_\_getitem\_\_

```python
def __getitem__(self, s: Union[int, tuple]) -> Union['Register', VirtualQubit]
```

```
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
```

<a id="stac.register.Register.__ge__"></a>

#### Register.\_\_ge\_\_

```python
def __ge__(self, other: 'Register') -> bool
```

```
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
```

<a id="stac.register.Register.append"></a>

#### Register.append

```python
def append(self, *registers: Union['Register', list['Register']]) -> None
```

```
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
```

<a id="stac.register.Register.num_qubits"></a>

#### Register.num\_qubits

```python
@property
def num_qubits(self) -> int
```

```
    Determine number of qubits in this Register recursively.
    
    Returns
    -------
    int
        Number of qubits.
```

<a id="stac.register.Register.structure"></a>

#### Register.structure

```python
def structure(self, depth: int = -1) -> None
```

```
    Print the register structure.
    
    Parameters
    ----------
    max_depth : int, optional
        Determine structure to this depth. The default is -1, which goes to
        max depth.
```

<a id="stac.register.Register.check_address"></a>

#### Register.check\_address

```python
def check_address(self, address: tuple) -> bool
```

```
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
```

<a id="stac.register.Register.qubit_addresses"></a>

#### Register.qubit\_addresses

```python
def qubit_addresses(
    self, my_address: Optional[tuple] = tuple()) -> list[tuple]
```

```
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
```

<a id="stac.register.Register.qubits"></a>

#### Register.qubits

```python
def qubits(self, register_type: Optional[str] = None) -> Iterator
```

```
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
```

<a id="stac.register.QubitRegister"></a>

## QubitRegister

```python
class QubitRegister(Register)
```

```
    Class to manipulate registers made out of virtual qubits.
```

<a id="stac.register.QubitRegister.__init__"></a>

#### QubitRegister.\_\_init\_\_

```python
def __init__(self,
             register_type: str,
             level: int,
             num_qubits: int,
             index: Optional[int] = None) -> None
```

```
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
```

<a id="stac.register.QubitRegister.copy"></a>

#### QubitRegister.copy

```python
def copy(self) -> 'QubitRegister'
```

```
    Create a copy of this register.
    
    Returns
    -------
    QubitRegister
        A copy of this register.
```

<a id="stac.register.QubitRegister.index"></a>

#### QubitRegister.index

```python
@property
def index(self) -> Union[int, None]
```

```
    Get index of this Register.
    
    Returns
    -------
    int
        Index of this Register. Is None if not set.
```

<a id="stac.register.QubitRegister.index"></a>

#### QubitRegister.index

```python
@index.setter
def index(self, value: int) -> None
```

```
    Set index of this Register.
    
    Also sets the assigned_register property of each qubit within this
    Register.
    
    Parameters
    ----------
    value : int
        Value to set.
```

<a id="stac.register.RegisterRegister"></a>

## RegisterRegister

```python
class RegisterRegister(Register)
```

```
    Class to manipulate registers made out of subregisters.
```

<a id="stac.register.RegisterRegister.__init__"></a>

#### RegisterRegister.\_\_init\_\_

```python
def __init__(self,
             register_type: str,
             level: int,
             subregisters: Optional[Iterator[Union[Register,
                                                   VirtualQubit]]] = None,
             code: Optional[Any] = None) -> None
```

```
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
```

<a id="stac.register.RegisterRegister.copy"></a>

#### RegisterRegister.copy

```python
def copy(self) -> 'RegisterRegister'
```

```
    Create a copy of this register.
    
    Returns
    -------
    RegisterRegister
        The copied register.
```

<a id="stac.register.RegisterRegister.index"></a>

#### RegisterRegister.index

```python
@property
def index(self) -> Optional[int]
```

```
    Get index of this Register.
    
    Returns
    -------
    int
        Index of this Register. Is None if not set.
```

<a id="stac.register.RegisterRegister.index"></a>

#### RegisterRegister.index

```python
@index.setter
def index(self, value: int) -> None
```

```
    Set index of this Register.
    
    Parameters
    ----------
    value : int
        Value to set.
```

<a id="stac.register.RegisterRegister.constituent_register_mapping"></a>

#### RegisterRegister.constituent\_register\_mapping

```python
def constituent_register_mapping(self) -> list[list]
```

```
    Determine the constituent register of every qubit within this Register.
    
    Returns
    -------
    list[list]
        List of lists, where each sublist is a
        [address, constituent_reigster] pair.
```

<a id="stac.concatenation.ConcatCode"></a>

## ConcatCode

```python
class ConcatCode(Code)
```

```
    Class to create concatenated codes.
```

<a id="stac.concatenation.ConcatCode.__init__"></a>

#### ConcatCode.\_\_init\_\_

```python
def __init__(self, *args: Any) -> None
```

```
    Construct a concatenated code.
    
    Parameters
    ----------
    *args:
        Can be on eof the following
    tuple[Code]
        A tuple of codes that will be concatenated in order.
    tuple[Code, int]
    The Code will be concatenated with itself.
    
    Raises
    ------
    TypeError
        DESCRIPTION.
```

<a id="stac.code.print_matrix"></a>

#### print\_matrix

```python
def print_matrix(array: Any, augmented: bool = False) -> None
```

```
    Display an array using latex.
    
    If augmented=True, then a line is placed
    in the center of the matrix, which is useful
    for printing the stabilizer generator matrix.
```

<a id="stac.code.print_paulis"></a>

#### print\_paulis

```python
def print_paulis(G: Any) -> None
```

```
    Print a set of Paulis as I,X,Y,Z.
```

<a id="stac.code.print_paulis_indexed"></a>

#### print\_paulis\_indexed

```python
def print_paulis_indexed(G: Any) -> None
```

```
    Print a set of Paulis as indexed X,Y,Z.
```

<a id="stac.code.Code"></a>

## Code

```python
class Code()
```

```
    Class for creating stabilizer codes.
```

<a id="stac.code.Code.__init__"></a>

#### Code.\_\_init\_\_

```python
def __init__(self, *args: Any) -> None
```

```
    Construct a stabilizer code.
    
    Parameters
    ----------
    There are multiple choices for construction. One choice is
    
    generator_matrix : numpy.array
        The Code is constructed using this generator matrix.
    
    Another option is,
    
    generators_x : numpy.array
    generators_z : numpy.array
        Pass two matrices of the same shape, that describe the X part and
        the Z part of the code.
```

<a id="stac.code.Code.__repr__"></a>

#### Code.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.code.Code.__str__"></a>

#### Code.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.code.Code.check_valid_code"></a>

#### Code.check\_valid\_code

```python
def check_valid_code(self) -> bool
```

```
    Check if code generators commute.
    
    Returns
    -------
    bool
        True if the code generators commute, false otherwise.
```

<a id="stac.code.Code.check_in_normalizer"></a>

#### Code.check\_in\_normalizer

```python
def check_in_normalizer(self, operator: Any) -> bool
```

```
    Check if an operator is in the normalizer of the stabilizer code.
    
    Checks if the operator commutes with every generator.
    
    Parameters
    ----------
    operator : numpy.array
        A 2n length numpy array of of the operator.
    
    Returns
    -------
    bool
        True if operator in normalizer, else False.
```

<a id="stac.code.Code.construct_standard_form"></a>

#### Code.construct\_standard\_form

```python
def construct_standard_form(self) -> (Any, Any, int)
```

```
    Construct the standard form a stabilizer matrix.
    
    Returns
    -------
    standard_generators_x: numpy.array
        The X part of the standard generator matrix.
    standard_generators_z: numpy.array
        The Z part of a standard generator matix.
    rankx: int
        The rank of the X part of the generator matrix..
```

<a id="stac.code.Code.construct_logical_operators"></a>

#### Code.construct\_logical\_operators

```python
def construct_logical_operators(self, method: str = "gottesman") -> (Any, Any)
```

```
    Construct logical operators for the code.
    
    Parameters
    ----------
    method : str
        Method to construct logical operators. Uses Gottesman's method by
        default.
    
    Returns
    -------
    logical_xs: numpy.array
        Array of logical xs. Each row is an operator.
    logical_zs: numpy.array
        Array of logical xs. Each row is an operator.
```

<a id="stac.code.Code.construct_logical_gate_circuits"></a>

#### Code.construct\_logical\_gate\_circuits

```python
def construct_logical_gate_circuits(self,
                                    syndrome_measurement_type: str = 'non_ft')
```

```
    Create the circuits that implement logical circuits for the code.
    
    Results are storted in logical_circuits.
    
    Parameters
    ----------
    syndrome_measurement_type: str
        Options are 'non_ft', 'cat'
```

<a id="stac.code.Code.find_destabilizers"></a>

#### Code.find\_destabilizers

```python
def find_destabilizers(self)
```

```
    Find the destabilizers of the standard form generators.
    
    Find the destabilizers of the standard form generators by exhaustive
    search. This will be slow for large codes but has the advantage that
    it will find the lowest weight destabilizers.
    
    Returns
    -------
    destab_gen_mat: numpy.array
        Array of shape m x 2n where each row is a destabilizer
```

<a id="stac.code.Code.construct_data_register"></a>

#### Code.construct\_data\_register

```python
def construct_data_register(self, level: int) -> RegisterRegister
```

```
    Create a data qubit register for this code.
    
    Parameters
    ----------
    level : int
        The concatenation level of the qubit.
    
    Returns
    -------
    RegisterRegister
        The data qubit register.
```

<a id="stac.code.Code.construct_syndrome_measurement_register"></a>

#### Code.construct\_syndrome\_measurement\_register

```python
def construct_syndrome_measurement_register(
        self,
        level: int,
        syndrome_measurement_type: str = 'non_ft') -> RegisterRegister
```

```
    Create a register appropriate for doing syndrome measurements.
    
    Parameters
    ----------
    level : int
        The concatenation level of the qubit.
    syndrome_measurement_type : str
        Options are 'non_ft',
                    'cat',
                    'cat_standard'.
        With the 'standard' postfix uses the standard form of the
        generators. 'non_ft' is Default.
    
    Returns
    -------
    RegisterRegister
        Register for encoded qubit.
```

<a id="stac.code.Code.construct_encoded_qubit_register"></a>

#### Code.construct\_encoded\_qubit\_register

```python
def construct_encoded_qubit_register(
        self,
        level: int,
        syndrome_measurement_type: str = 'non_ft') -> RegisterRegister
```

```
    Create a register appropriate for creating an encoded qubit.
    
    Parameters
    ----------
    level : int
        The concatenation level of the qubit.
    syndrome_measurement_type : str
        Options are 'non_ft',
                    'cat',
                    'cat_standard'.
        With the 'standard' postfix uses the standard form of the
        generators. 'non_ft' is Default.
    
    Returns
    -------
    RegisterRegister
        Register for encoded qubit.
```

<a id="stac.code.Code.construct_encoding_circuit"></a>

#### Code.construct\_encoding\_circuit

```python
def construct_encoding_circuit(self,
                               syndrome_measurement_type: str = 'none'
                               ) -> Circuit
```

```
    Construct an encoding circuit for the code using Gottesman's method.
    
    Parameters
    ----------
    syndrome_measurement_type : str, optional
        Possible types are
            * 'none': Creates a simple data register only. Default.
            * 'non_ft',
            * 'cat',
            * 'cat_standard'.
            With the 'standard' postfix uses the standard form of the
            generators.
        The syndrome registers will be empty, but useful if the circuit
        is part of a larget circuit.
    
    Returns
    -------
    encoding_circuit : Circuit
        The encoding circuit.
```

<a id="stac.code.Code.construct_decoding_circuit"></a>

#### Code.construct\_decoding\_circuit

```python
def construct_decoding_circuit(self) -> Circuit
```

```
    Construct an decoding circuit for the code using Gottesman's method.
    
    Returns
    -------
    decoding_circuit : Circuit
        The decoding circuit.
```

<a id="stac.code.Code.construct_syndrome_circuit"></a>

#### Code.construct\_syndrome\_circuit

```python
def construct_syndrome_circuit(self,
                               syndrome_measurement_type: str = 'non_ft',
                               assign_circuit: bool = True) -> Circuit
```

```
    Construct a circuit to measure the stabilizers of the code.
    
    ----------
    syndrome_measurement_type : str
        Options are 'non_ft',
                    'non_ft_standard',
                    'cat',
                    'cat_standard'.
        With the 'standard' postfix uses the standard form of the
        generators. If no argument, then 'non_ft' is Default.
    assign_circuit : bool, optional
        If true, the circuit is assigned to self.syndrome_circuit. The
        default is True.
    
    Returns
    -------
    syndrome_circuit : Circuit
        The circuit for measuring the stabilizers.
```

<a id="stac.code.Code.construct_encoded_qubit"></a>

#### Code.construct\_encoded\_qubit

```python
def construct_encoded_qubit(self,
                            J: int,
                            syndrome_measurement_type: str = 'non_ft'
                            ) -> Circuit
```

```
    Create an encoded qubit at the Jth concatenation level.
    
    Parameters
    ----------
    J : int
        Concatenation level.
    syndrome_measurement_type : str, optional
        Options are 'non_ft',
                    'non_ft_standard',
                    'cat',
                    'cat_standard'.
        With the 'standard' postfix uses the standard form of the
        generators. If no argument, then 'non_ft' is Default.
    
    Returns
    -------
    Circuit
        DESCRIPTION.
```

<a id="stac.instructionblock.InstructionBlock"></a>

## InstructionBlock

```python
class InstructionBlock()
```

```
    Class for creating and manipulating blocks of circuit instructions.
```

<a id="stac.instructionblock.InstructionBlock.__repr__"></a>

#### InstructionBlock.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the block.
```

<a id="stac.instructionblock.InstructionBlock.__str__"></a>

#### InstructionBlock.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the block.
```

<a id="stac.instructionblock.InstructionBlock.__iter__"></a>

#### InstructionBlock.\_\_iter\_\_

```python
def __iter__(self) -> Iterator
```

```
    Return iterator of the block.
```

<a id="stac.instructionblock.InstructionBlock.__getitem__"></a>

#### InstructionBlock.\_\_getitem\_\_

```python
def __getitem__(self, ind) -> Union[Any, list[Any]]
```

```
    Make Timepoint subscriptable.
```

<a id="stac.instructionblock.InstructionBlock.__len__"></a>

#### InstructionBlock.\_\_len\_\_

```python
def __len__(self) -> int
```

```
    Return number of operations in the block.
```

<a id="stac.instructionblock.InstructionBlock.insert"></a>

#### InstructionBlock.insert

```python
def insert(self, i, ins) -> int
```

```
    Insert instruction at particular index.
```

<a id="stac.instructionblock.InstructionBlock.copy"></a>

#### InstructionBlock.copy

```python
def copy(self) -> 'InstructionBlock'
```

```
    Return a copy of the block.
```

<a id="stac.instructionblock.InstructionBlock.append"></a>

#### InstructionBlock.append

```python
def append(self, obj) -> None
```

```
    Append object.
```

<a id="stac.instructionblock.AnnotationBlock"></a>

## AnnotationBlock

```python
class AnnotationBlock(InstructionBlock)
```

```
    Class to create blocks that hold annotations.
```

<a id="stac.instructionblock.RepetitionBlock"></a>

## RepetitionBlock

```python
class RepetitionBlock(InstructionBlock)
```

```
    Class to create blocks of repeating instructions.
```

<a id="stac.instructionblock.IfBlock"></a>

## IfBlock

```python
class IfBlock(InstructionBlock)
```

```
    Class to store conditional instructions.
```

<a id="stac.qubit.PhysicalQubit"></a>

## PhysicalQubit

```python
class PhysicalQubit()
```

```
    Class to create and manipulate physical qubits.
```

<a id="stac.qubit.PhysicalQubit.__init__"></a>

#### PhysicalQubit.\_\_init\_\_

```python
def __init__(self, index: int, coordinates: Union[int, tuple],
             interactable_qubits: list[Union[int, tuple]]) -> None
```

```
    Construct a physical qubit.
    
    Parameters
    ----------
    index : int
        Index of qubits within its Register.
    coordinates : Union[int, tuple]
        The coordinate of the qubit.
    interactable_qubits : list[Union[int, tuple]]
        The qubits this qubit can interact with.
```

<a id="stac.qubit.VirtualQubit"></a>

## VirtualQubit

```python
class VirtualQubit()
```

```
    Class to create and manipulate virtual qubits.
```

<a id="stac.qubit.VirtualQubit.__init__"></a>

#### VirtualQubit.\_\_init\_\_

```python
def __init__(self,
             level: int,
             index_in_assigned_register: int,
             assigned_register: tuple = None,
             index_in_constituent_register: int = None,
             constituent_register: tuple = None) -> None
```

```
    Construct a virtual qubit.
    
    Parameters
    ----------
    level : int
        The level of the Circuit this qubit is at.
    index_in_assigned_register : int
        The index within its assigned register.
    assigned_register : tuple, optional
        The address of the Register this qubit is part of. The default is
        None.
    index_in_constituent_register : int, optional
        The index within its constituent register. The default is None.
    constituent_register : tuple, optional
        Encoded qubits at level > 1 are made of a Register. This points to
        the address of that Register. The default is None.
```

<a id="stac.qubit.VirtualQubit.index_in_assigned_register"></a>

#### VirtualQubit.index\_in\_assigned\_register

```python
@property
def index_in_assigned_register(self) -> int
```

```
    Get index in assigned register.
    
    Returns
    -------
    int
        Index in assigned register.
```

<a id="stac.qubit.VirtualQubit.index_in_assigned_register"></a>

#### VirtualQubit.index\_in\_assigned\_register

```python
@index_in_assigned_register.setter
def index_in_assigned_register(self, value: int) -> None
```

```
    Set index in assigned register.
    
    Parameters
    ----------
    value : int
        Value to set.
```

<a id="stac.qubit.VirtualQubit.index"></a>

#### VirtualQubit.index

```python
@property
def index(self)
```

```
    Get index in assigned register.
    
    Returns
    -------
    int
        Index in assigned register.
```

<a id="stac.qubit.VirtualQubit.index"></a>

#### VirtualQubit.index

```python
@index.setter
def index(self, value: int) -> None
```

```
    Set index in assigned register.
    
    Parameters
    ----------
    value : int
        Value to set.
```

<a id="stac.qubit.VirtualQubit.copy"></a>

#### VirtualQubit.copy

```python
def copy(self) -> 'VirtualQubit'
```

```
    Create copy of this register.
    
    Returns
    -------
    VirtualQubit
        The copy of self.
```

<a id="stac.timepoint.Timepoint"></a>

## Timepoint

```python
class Timepoint()
```

```
    Class to create and manipulate timepoints.
```

<a id="stac.timepoint.Timepoint.__init__"></a>

#### Timepoint.\_\_init\_\_

```python
def __init__(self, new_op: Operation = None) -> None
```

```
    Construct a Timepoint.
    
    Parameters
    ----------
    new_op : Operation, optional
        This operation will be appended to the Timepoint. The default is
        None.
```

<a id="stac.timepoint.Timepoint.__repr__"></a>

#### Timepoint.\_\_repr\_\_

```python
def __repr__(self) -> str
```

```
    Return a representation of the object.
```

<a id="stac.timepoint.Timepoint.__str__"></a>

#### Timepoint.\_\_str\_\_

```python
def __str__(self) -> str
```

```
    Return a string representation of the object.
```

<a id="stac.timepoint.Timepoint.__iter__"></a>

#### Timepoint.\_\_iter\_\_

```python
def __iter__(self) -> Iterator[Operation]
```

```
    Return iterator of the Timepoint.
```

<a id="stac.timepoint.Timepoint.__getitem__"></a>

#### Timepoint.\_\_getitem\_\_

```python
def __getitem__(self, ind) -> Union[Operation, list[Operation]]
```

```
    Make Timepoint subscriptable.
```

<a id="stac.timepoint.Timepoint.__len__"></a>

#### Timepoint.\_\_len\_\_

```python
def __len__(self) -> int
```

```
    Return number of operations in the Timepoint.
```

<a id="stac.timepoint.Timepoint.copy"></a>

#### Timepoint.copy

```python
def copy(self) -> 'Timepoint'
```

```
    Return a copy of the Timepoint.
```

<a id="stac.timepoint.Timepoint.append"></a>

#### Timepoint.append

```python
def append(self, new_op: Operation) -> None
```

```
    Append operation to this Timepoint.
    
    Parameters
    ----------
    new_op : Operation
        Operation to append.
    
    Raises
    ------
    Exception
        If new_op can't be appended to current Timepoint.
```

<a id="stac.timepoint.Timepoint.can_append"></a>

#### Timepoint.can\_append

```python
def can_append(self, new_op: Operation) -> bool
```

```
    Check if an Operation can be appended to this Timepoint.
    
    Parameters
    ----------
    new_op : Operation
        Operation to be checked.
    
    Returns
    -------
    bool
        True if Operation can be appended, otherwise False.
```

<a id="stac.timepoint.Timepoint.rebase_qubits"></a>

#### Timepoint.rebase\_qubits

```python
def rebase_qubits(self, new_base: tuple) -> 'Timepoint'
```

```
    Create Timepoint with new base address for all controls and targets.
    
    Parameters
    ----------
    new_base : tuple
        New base address. Must have length smaller than the shortest
        address within all controls and targets within qubits.
    
    Returns
    -------
    tp : Timepoint
        Timepoint with new base address.
```

<a id="stac.timepoint.Timepoint.can_add"></a>

#### Timepoint.can\_add

```python
def can_add(self, other: 'Timepoint') -> bool
```

```
    Check if a Timepoint can be added to this Timepoint.
    
    Parameters
    ----------
    other : Timepoint
        The Timepoint to be checked.
    
    Returns
    -------
    bool
        True if other can be added, otherwise False.
```

<a id="stac.timepoint.Timepoint.__add__"></a>

#### Timepoint.\_\_add\_\_

```python
def __add__(self, other: 'Timepoint') -> 'Timepoint'
```

```
    Create Timepoint that is sum of other Timepoint and this Timepoint.
    
    Parameters
    ----------
    other : Timepoint
        Timepoint to be added.
    
    Returns
    -------
    tp : Timepoint
        DESCRIPTION.
```

<a id="stac.timepoint.Timepoint.__iadd__"></a>

#### Timepoint.\_\_iadd\_\_

```python
def __iadd__(self, other: 'Timepoint') -> 'Timepoint'
```

```
    Add other Timepoint to this Timepoint.
    
    Parameters
    ----------
    other : Timepoint
        Timepoint to be added.
    
    Returns
    -------
    Timepoint
        Summed Timepoints.
```

<a id="stac.commoncodes.CommonCodes"></a>

## CommonCodes

```python
class CommonCodes()
```

```
    Class to provide some common codes.
```

<a id="stac.commoncodes.CommonCodes.__init__"></a>

#### CommonCodes.\_\_init\_\_

```python
def __init__(self)
```

```
    Use the generate_code method to create codes.
```

<a id="stac.commoncodes.CommonCodes.generate_code"></a>

#### CommonCodes.generate\_code

```python
@classmethod
def generate_code(cls, codename: str) -> Code
```

```
    Generate an internally stored Code.
    
    Parameters
    ----------
    codename : str
        Can be one of the following.
            * [[7,1,3]]
            * [[5,1,3]]
            * [[4,2,2]]
            * [[8,3,3]]
            * [[6,4,2]]
    
    Raises
    ------
    Exception
        If codename is not recognized.
    
    Returns
    -------
    Code
        The corresponding code.

```