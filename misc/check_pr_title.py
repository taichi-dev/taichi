import sys, os

title = sys.argv[1]
print(f'Checking PR title: {title}')

prtags = []
this_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_dir, 'prtags.txt')) as f:
    for line in f.readlines():
        try:
            tag, desc = line.strip().split(maxsplit=1)
            prtags.append(tag)
        except ValueError:
            pass

if not title.startswith('['):
    raise Exception(f'PR tltle does not starts with any tag: {title}')

if title.endswith(' '):
    raise Exception(f'PR tltle should not ends with a space: {title}')

if '`' in title:
    raise Exception(f'PR tltle should not contain backquotes (`): {title}')

for x in title.split(']')[1:]:
    if x[0] != ' ':
        raise Exception(f'No space before: {x}')
    if x[1] == ' ':
        raise Exception(f'Extra space before: {x[2:]}')

x = title.split(']')[-1].strip()
if x[0].islower():
    raise Exception(f'PR title should be uppercase at: {x}')

for x in title.split('] ')[:-1]:
    if x[0] != '[':
        raise Exception(f'No starting [ for tag: {x}]')
    if x[1:].lower() not in prtags:
        raise Exception(f'Unrecognized PR tag: [{x[1:]}]')

print('OK!')
