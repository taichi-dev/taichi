import sys, os, json
from colorama import Fore, Style

title = sys.argv[1]
print(f'Checking PR title: {Fore.CYAN + title + Fore.RESET}')

this_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(this_dir, 'prtags.json')
with open(json_path) as f:
    prtags = json.load(f)

def error(msg):
    print(Style.BRIGHT + Fore.YELLOW + msg + Style.RESET_ALL)
    exit(1)

if not title.startswith('['):
    error(f'PR title does not start with any tag: {title}')

if title.endswith(' '):
    error(f'PR title should not end with a space: {title}')

if title.endswith(']'):
    error(f'PR title should have bodies regardless tags: {title}')

if '`' in title:
    error(f'PR title should not contain backquotes (`): {title}')

for x in title.split(']')[1:]:
    if x[0] != ' ':
        error(f'No space before: {x}')
    if x[1] == ' ':
        error(f'Extra space before: {x[2:]}')

x = title.split(']')[-1].strip()
if x[0].islower():
    error(f'PR title should be uppercase at: {x}')

had_upper = False
for x in title.split('] ')[:-1]:
    if x[0] != '[':
        error(f'No starting [ for tag: {x}]')
    if x[1:].lower() not in prtags.keys():
        error(f'Unrecognized PR tag: [{x[1:]}]')
    # 'Misc'.islower() -> False, 'Misc'.isupper() -> False
    # 'misc'.islower() -> True, 'misc'.isupper() -> False
    if not x[1:].islower():
        if had_upper:
            error(f'At most 1 uppercase tag expected, got: [{x[1:]}]')
        had_upper = True

print('OK!')
