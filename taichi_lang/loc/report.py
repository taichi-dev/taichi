import os
import pathlib

for filename in pathlib.Path('src').glob('**/*.c'):
    print(filename)

def account(folder):
    total_loc = 0
    for fn in pathlib.Path(folder).glob('**/*'):
        if os.path.isdir(fn):
            continue
        os.system('clang-format-6.0 -i {}'.format(fn))
        with open(fn) as f:
            total_loc += len(f.readlines())
    print(folder, total_loc)

for f in sorted(os.listdir('.')):
    if os.path.isdir(f):
        account(f)
