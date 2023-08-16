"""
The operations supported by stac.

ins_type: int
    Instruction type. 0 for operation, 1 for annotation
num_targets: int
    If positive, indicates the exact number of targets. If -1, the
    instruction can take any number of targets.
"""

_operations = dict()

for name in ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CAT']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
        'qasm_str': name.lower() + ' q[{t0}];\n'
        }
_operations['I']['qasm_str'] = 'id q[{t0}];\n'
_operations['CAT']['stim_str'] = 'H'
_operations['CAT']['draw_text'] = 'H'
_operations['CAT']['draw_img'] = 'H'

for name in ['RX', 'RY', 'RZ']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': True,
        'num_targets': 1,
        'num_parameters': 1,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
        'qasm_str': name.lower() + '({p0}) q[{t0}];\n'
        }

for name in ['CX', 'CY', 'CZ']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 2,
        'draw_text': ['●', name[1]],
        'draw_img': ['●', name[1]],
        'stim_str': name,
        'qasm_str': name.lower() + ' q[{t0}],q[{t1}];\n'
        }
_operations['CX']['draw_text'][1] = '⊕'
_operations['CZ']['draw_text'][1] = '●'

for name in ['R', 'M', 'MR']:
    _operations[name] = {
        'ins_type': 0,
        'is_parameterized': False,
        'num_targets': 1,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
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
        'stim_str': name,
        'qasm_str': ''
        }
_operations['TICK']['qasm_str'] = 'barrier q;\n'
