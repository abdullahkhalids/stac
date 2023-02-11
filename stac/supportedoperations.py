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

_operations = dict()
for name in ['X', 'Y', 'Z', 'H', 'S', 'T']:
    _operations[name] = {
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name]
        }

for name in ['CX', 'CY', 'CZ']:
    _operations[name] = {
        'is_parameterized': False,
        'num_targets': 2,
        'draw_text': ['●', name[1]],
        'draw_img': ['●', name[1]]
        }
_operations['CX']['draw_text'][1] = '⊕'
_operations['CY']['draw_text'][0] = '●'

for name in ['R', 'M', 'MR']:
    _operations[name] = {
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name]
        }
_operations['MR']['draw_text'][0] = 'm'

for name in ['TICK']:
    _operations[name] = {
        'is_parameterized': False,
        'num_targets': 0,
        }

# num_parameters