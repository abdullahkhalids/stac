#!/bin/python
import subprocess

filename = "api_reference.md"

config = '''{
'loaders': [{'type': 'python'}],
'processors': [
    {'type': 'filter',
     'expression': 'not name in ["stac", "stac.supportedoperations"] and default()',
     'do_not_filter_modules': False
     },
    {'type': 'smart'},
    {'type': 'crossref'}],
'renderer': {
    'type': 'markdown',
    'render_toc': True,
    'render_toc_title': "Api Reference",
    'add_module_prefix': False,
    'add_full_prefix': False,
    'render_module_header': False,
    'docstrings_as_blockquote': True,
    'add_method_class_prefix': True,
    'descriptive_class_title': False
    }
}
'''

output = subprocess.run(['pydoc-markdown',
                         '-I', '.',
                         config],
                        capture_output=True)

so = output.stdout.decode()
L = so.splitlines(True)

b = False
i = 0
while i < len(L):
    line = L[i]
    if line == '```\n' and not b:
        L[i+1] += '```\n'
        b = True
    elif line[0:2] == '<a' and b:
        L[i-1] = '```\n\n'
        b = False
    i += 1

L.append('```')

importstac = '''```python
>>> import stac

```

'''

for i, line in enumerate(L):
    if line[0:2] != '<a':
        continue
    else:
        L.insert(i, importstac)
        break

with open(filename, "w") as outfile:
    for line in L:
        outfile.write(line)
