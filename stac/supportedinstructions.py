"""
The instructions supported by stac.

ins_type: int
    Instruction type. 0 for operation, 1 for annotation
num_targets: int
    If positive, indicates the exact number of targets. If -1, the
    instruction can take any number of targets.
"""

instructions = dict()

for name in ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'CAT']:
    instructions[name] = {
        'ins_type': 0,
        'num_targets': 1,
        'control_targets': set(),
        'num_parameters': 0,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
        'qasm_str': name.lower() + ' q[{t0}];\n'
        }
instructions['I']['qasm_str'] = 'id q[{t0}];\n'
instructions['CAT']['stim_str'] = 'H'
instructions['CAT']['draw_text'] = 'H'
instructions['CAT']['draw_img'] = 'H'

for name in ['RX', 'RY', 'RZ']:
    instructions[name] = {
        'ins_type': 0,
        'num_targets': 1,
        'control_targets': set(),
        'num_parameters': 1,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
        'qasm_str': name.lower() + '({p0}) q[{t0}];\n'
        }

for name in ['CX', 'CY', 'CZ']:
    instructions[name] = {
        'ins_type': 0,
        'num_targets': 2,
        'control_targets': {0},
        'num_parameters': 0,
        'draw_text': ['●', name[1]],
        'draw_img': ['●', name[1]],
        'stim_str': name,
        'qasm_str': name.lower() + ' q[{t0}],q[{t1}];\n'
        }
instructions['CX']['draw_text'][1] = '⊕'
instructions['CZ']['draw_text'][1] = '●'

for name in ['R', 'M', 'MR']:
    instructions[name] = {
        'ins_type': 0,
        'num_targets': 1,
        'control_targets': set(),
        'num_parameters': 0,
        'draw_text': [name],
        'draw_img': [name],
        'stim_str': name,
        'qasm_str': ''
        }
instructions['MR']['draw_text'][0] = 'm'
instructions['R']['qasm_str'] = 'reset q[{t0}];\n'
instructions['M']['qasm_str'] = 'measure q[{t0}] -> c[{t0}];\n'
instructions['MR']['qasm_str'] = 'measure q[{t0}] -> c[{t0}];\n'

# annotations
instructions['TICK'] = {
    'ins_type': 1,
    'num_targets': 0,
    'control_targets': set(),
    'num_parameters': 0,
    'stim_str': name,
    'qasm_str': 'barrier q;\n'
    }

instructions['DETECTOR'] = {
    'ins_type': 1,
    'num_targets': -1,
    'control_targets': set(),
    'num_parameters': 0,
    'draw_text': None,
    'draw_img': None,
    'stim_str': None,
    'qasm_str': None
    }
