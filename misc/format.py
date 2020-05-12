#!/usr/bin/python3

import os, re
import subprocess as sp
from pathlib import Path
from colorama import Fore, Back, Style
from yapf.yapflib.yapf_api import FormatFile
from git import Repo

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def has_suffix(f, suffixes):
    for suf in suffixes:
        if f.endswith('.' + suf):
            return True
    return False


def format_plain_text(fn):
    formatted = ''
    with open(fn, 'r') as f:
        for l in f:
            l = l.rstrip()
            if l.find('\t') != -1:
                print(f'Warning: found tab in {fn}. Skipping...')
                return
            formatted += l + '\n'
    while len(formatted) and formatted[-1] == '\n':
        formatted = formatted[:-1]
    formatted += '\n'
    with open(fn, 'w') as f:
        f.write(formatted)


def _find_clang_format_bin():
    candidates = ['clang-format-6.0', 'clang-format']
    result = None

    for c in candidates:
        try:
            if sp.run([c, '--version'], stdout=sp.DEVNULL,
                      stderr=sp.DEVNULL).returncode == 0:
                result = c
                break
        except:
            pass
    if result is None:
        print(Fore.YELLOW +
              'Did not find any clang-format executable, skipping C++ files',
              file=sys.stderr)
    else:
        print('C++ formatter: {}{}'.format(Fore.GREEN, result))
    print(Style.RESET_ALL)
    return result


def main(all=False, diff=None):
    repo = Repo(repo_dir)

    if all:
        directories = [
            'taichi', 'tests', 'examples', 'misc', 'python', 'benchmarks',
            'docs', 'misc'
        ]
        files = []
        for d in directories:
            files += list(Path(os.path.join(repo_dir, d)).rglob('*'))
    else:
        if diff is None:

            def find_diff_or_empty(s):
                try:
                    return repo.index.diff(s)
                except:
                    return []

            # TODO(#628): Have a way to customize the repo names, in order to
            # support noncanonical namings.
            #
            # Finds all modified files from upstream/master to working tree
            # 1. diffs between the index and upstream/master. Also inclulde
            # origin/master for repo owners.
            files = find_diff_or_empty('upstream/master')
            files += find_diff_or_empty('origin/master')
            # 2. diffs between the index and the working tree
            # https://gitpython.readthedocs.io/en/stable/tutorial.html#obtaining-diff-information
            files += repo.index.diff(None)
        else:
            files = repo.index.diff(diff)
        files = list(map(lambda x: os.path.join(repo_dir, x.a_path), files))

    files = sorted(set(map(str, files)))
    clang_format_bin = _find_clang_format_bin()
    print('Code formatting ...')
    for fn in files:
        if not os.path.exists(fn):
            continue
        if os.path.isdir(fn):
            continue
        if fn.find('.pytest_cache') != -1:
            continue
        if fn.find('docs/build/') != -1:
            continue
        if re.match(r'.*examples\/[a-z_]+\d\d+\.py$', fn):
            print(f'Skipping example file {fn}...')
            continue
        if fn.endswith('.py'):
            print('Formatting "{}"'.format(fn))
            FormatFile(fn,
                       in_place=True,
                       style_config=os.path.join(repo_dir, 'misc',
                                                 '.style.yapf'))
        elif clang_format_bin and has_suffix(fn, ['cpp', 'h', 'cu', 'cuh']):
            print('Formatting "{}"'.format(fn))
            os.system('{} -i -style=file {}'.format(clang_format_bin, fn))
        elif has_suffix(fn, ['txt', 'md', 'rst', 'cfg', 'll', 'ptx']):
            print('Formatting "{}"'.format(fn))
            format_plain_text(fn)
        elif has_suffix(fn, [
                'pyc', 'png', 'jpg', 'bmp', 'gif', 'gitignore', 'whl', 'mp4',
                'html'
        ]):
            pass
        else:
            print(f'Skipping {fn}...')

    print('Formatting done!')


if __name__ == '__main__':
    main()
