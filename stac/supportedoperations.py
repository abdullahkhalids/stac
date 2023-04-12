"""
The operations supported by stac.

ins_type: int
    Instruction type. 0 for operation, 1 for annotation
num_targets: int
    If positive, indicates the exact number of targets. If -1, the
    instruction can take any number of targets.
"""

_single_qubit_gates = {'I', 'X', 'Y', 'Z', 'H', 'S', 'T'}
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
for name in ['I', 'X', 'Y', 'Z', 'H', 'S', 'T']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name],
        'qasm_str': name.lower() + ' q[{t0}];\n'
        }
_operations['I']['qasm_str'] = 'id q[{t0}];\n'

for name in ['CX', 'CY', 'CZ']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 2,
        'draw_text': ['●', name[1]],
        'draw_img': ['●', name[1]],
        'qasm_str': name.lower() + ' q[{t0}],q[{t1}];\n'
        }
_operations['CX']['draw_text'][1] = '⊕'
_operations['CZ']['draw_text'][0] = '●'

for name in ['R', 'M', 'MR']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name],
        'qasm_str': ''
        }
_operations['MR']['draw_text'][0] = 'm'
_operations['R']['qasm_str'] = 'reset q[{t0}];\n'
_operations['M']['qasm_str'] = 'measure q[{t0}] -> c[{t0}];\n'
_operations['MR']['qasm_str'] = 'measure q[{t0}] -> c[{t0}];\n'


for name in ['TICK']:
    _operations[name] = {
        'ins_type': 1,
        'is_parameterized': False,
        'num_targets': 0,
        }
_operations['TICK']['qasm_str'] = 'barrier;\n'

