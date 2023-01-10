"""The operations supported by stac."""

_single_qubit_gates = {'X', 'Y', 'Z', 'H', 'S', 'T'}
_two_qubit_gates = {'CX', 'CY', 'CZ'}
_measurements = {'R', 'M', 'MR'}
_circuit_annotations = {'TICK'}

_zero_qubit_operations = _circuit_annotations
_one_qubit_operations = set.union(_single_qubit_gates,
                                  _measurements)
_two_qubit_operations = _two_qubit_gates

_quantum_operations = set.union(
    _single_qubit_gates,
    _two_qubit_gates,
    _measurements,
)

_circuit_operations = set.union(
    _quantum_operations,
    _circuit_annotations
)
