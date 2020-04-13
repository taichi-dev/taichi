from git import Repo

# Usage: python3 misc/make_changelog.py 0.5.9

import sys

ver = sys.argv[1]

g = Repo('.')
commits = list(g.iter_commits('master', max_count=200))
begin, end = -1, 0


def format(c):
    return f'{c.summary} (by **{c.author}**)'


print('Notable changes:')

notable_changes = {}
all_changes = []

details = {
    'cpu': 'CPU backends',
    'cuda': 'CUDA backend',
    'doc': 'Documentation',
    'infra': 'Infrastructure',
    'ir': 'Intermediate representation',
    'lang': 'Language and syntax',
    'metal': 'Metal backend',
    'opengl': 'OpenGL backend',
    'misc': 'Miscellaneous',
    'opt': 'Optimization',
    'example': 'Examples',
    'pypi': 'PyPI package',
    'autodiff': 'Automatic differentiation',
    'gui': 'GUI',
    'llvm': 'LLVM backend (CPU and CUDA)',
}

print(f'- (, 2020) v{ver} released')
for i, c in enumerate(commits):
    s = format(c)
    if s.startswith('[release]'):
        if ver in s:
            all_changes = []
            notable_changes = {}
        else:
            break
    tags = []
    while s[0] == '[':
        r = s.find(']')
        tag = s[1:r]
        tags.append(tag)
        s = s[r + 1:]
        s = s.strip()
    for tag in tags:
        if tag[0].isupper():
            tag = tag.lower()
            if tag not in notable_changes:
                notable_changes[tag] = []
            notable_changes[tag].append(s)
    if s.startswith('[release]'):
        break
    all_changes.append(format(c))

for tag in sorted(notable_changes.keys()):
    print(f'   - **{details[tag]}**')
    for item in notable_changes[tag]:
        print(f'      - {item}')
print(
    f'   - [Full log](https://github.com/taichi-dev/taichi/releases/tag/{ver})'
)
print()

print('Full changelog:')
for c in all_changes:
    print(f'   - {c}')
