#!/usr/bin/python3

import os
import re
import subprocess as sp
import sys
from pathlib import Path

from colorama import Back, Fore, Style
from git import Repo
from yapf.yapflib.yapf_api import FormatFile

_has_isort = False
try:
    import isort
    _has_isort = True
except ImportError:
    # TODO(#2223): Make `isort` a required package in a future release
    print('Please install `isort` or the formatter may not work')

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_yapf_config_path = os.path.join(repo_dir, 'misc', '.style.yapf')


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
            formatted += l + '\n'
    while len(formatted) and formatted[-1] == '\n':
        formatted = formatted[:-1]
    formatted += '\n'
    with open(fn, 'w') as f:
        f.write(formatted)


def find_clang_format_bin():
    try:
        return find_clang_format_bin.clang_format_bin
    except AttributeError:
        pass

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
    find_clang_format_bin.clang_format_bin = result
    return result


def format_py_file(filename):
    FormatFile(filename, in_place=True, style_config=_yapf_config_path)
    if _has_isort:
        isort.file(filename)
    format_plain_text(filename)


def main(all=False, diff=None):
    repo = Repo(repo_dir)

    if all:
        directories = [
            'taichi',
            'tests',
            'examples',
            'misc',
            'python',
            'benchmarks',
            'docs',
            'cmake',
        ]
        files = list(Path(repo_dir).glob(
            '*'))  # Include all files under the root folder
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
            print(f'Skipping example file "{fn}"...')
            continue
        if not format_file(fn):
            print(f'Skipping "{fn}"...')

    print('Formatting done!')


def format_file(fn):
    clang_format_bin = find_clang_format_bin()
    print('Formatting "{}"'.format(fn))
    if fn.endswith('.py'):
        format_py_file(fn)
        return True
    elif clang_format_bin and has_suffix(fn, ['cpp', 'h', 'c', 'cu', 'cuh']):
        os.system('{} -i -style=file {}'.format(clang_format_bin, fn))
        format_plain_text(fn)
        return True
    elif has_suffix(fn, [
            'txt', 'md', 'rst', 'cfg', 'ini', 'map', 'cmake'
    ]) or (os.path.basename(fn)[0].isupper()
           and fn.endswith('file')):  # E.g., Dockerfile and Jenkinsfile
        format_plain_text(fn)
        return True
    else:
        return False


if __name__ == '__main__':
    main()
