# Usage: python3 misc/make_changelog.py [v0.x.y]

import sys
from git import Repo


def main(ver='master', repo_dir='.'):
    g = Repo(repo_dir)
    commits = list(g.iter_commits(ver, max_count=200))
    begin, end = -1, 0

    def format(c):
        return f'{c.summary} (by **{c.author}**)'

    notable_changes = {}
    all_changes = []

    details = {
        'cpu': 'CPU backends',
        'cuda': 'CUDA backend',
        'doc': 'Documentation',
        'infra': 'Infrastructure',
        'cli': 'Command line interface',
        'ir': 'Intermediate representation',
        'lang': 'Language and syntax',
        'metal': 'Metal backend',
        'opengl': 'OpenGL backend',
        'misc': 'Miscellaneous',
        'std': 'Standard library',
        'opt': 'IR optimization passes',
        'example': 'Examples',
        'pypi': 'PyPI package',
        'autodiff': 'Automatic differentiation',
        'sparse': 'Sparse computation',
        'gui': 'GUI',
        'llvm': 'LLVM backend (CPU and CUDA)',
        'refactor': 'Refactor',
        'bug': 'Bug fixes',
        'test': 'Tests',
        'benchmark': 'Benchmarking',
        'async': 'AsyncEngine',
        'workflow': 'GitHub Actions/Workflows',
        'linux': 'Linux',
        'mac': 'Mac OS X',
        'windows': 'Windows',
        'perf': 'Performance improvements',
        'release': '',
    }

    for i, c in enumerate(commits):
        s = format(c)
        if s.startswith('[release]'):
            if i == 0:
                continue
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
            if tag.lower() in details:
                if details[tag.lower()] == '':
                    # E.g. 'release' does not need to appear in the change log
                    continue
                if tag[0].isupper():
                    tag = tag.lower()
                    if tag not in notable_changes:
                        notable_changes[tag] = []
                    notable_changes[tag].append(s)
            else:
                print(
                    f'** Warning: tag {tag.lower()} undefined in the "details" dict. Please include the tag into "details", unless the tag is a typo.'
                )

        all_changes.append(format(c))

    res = 'Highlights:\n'
    for tag in sorted(notable_changes.keys()):
        res += f'   - **{details[tag]}**\n'
        for item in notable_changes[tag]:
            res += f'      - {item}\n'

    res += '\nFull changelog:\n'
    for c in all_changes:
        res += f'   - {c}\n'

    return res


if __name__ == '__main__':
    ver = sys.argv[1] if len(sys.argv) > 1 else 'master'
    res = main(ver)
    print(res)
