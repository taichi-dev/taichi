import sys, os, json

title = sys.argv[1]
print(f'Checking PR title: {title}')

this_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(this_dir, 'prtags.json')
with open(json_path) as f:
    prtags = json.load(f)

if not title.startswith('['):
    exit(f'PR title does not start with any tag: {title}')

if title.endswith(' '):
    exit(f'PR title should not end with a space: {title}')

if title.endswith(']'):
    exit(f'PR title should have bodies regardless tags: {title}')

if '`' in title:
    exit(f'PR title should not contain backquotes (`): {title}')

for x in title.split(']')[1:]:
    if x[0] != ' ':
        exit(f'No space before: {x}')
    if x[1] == ' ':
        exit(f'Extra space before: {x[2:]}')

x = title.split(']')[-1].strip()
if x[0].islower():
    exit(f'PR title should be uppercase at: {x}')

had_upper = False
for x in title.split('] ')[:-1]:
    if x[0] != '[':
        exit(f'No starting [ for tag: {x}]')
    if x[1:].lower() not in prtags.keys():
        exit(f'Unrecognized PR tag: [{x[1:]}]')
    # 'Misc'.islower() -> False, 'Misc'.isupper() -> False
    # 'misc'.islower() -> True, 'misc'.isupper() -> False
    if not x[1:].islower():
        if had_upper:
            exit(f'At most 1 uppercase tag expected, got: [{x[1:]}]')
        had_upper = True

print('OK!')
