# -*- coding: utf-8 -*-

# -- stdlib --
import importlib
import os
import sys
from pathlib import Path

# -- third party --
# -- own --


# -- code --
def is_in_venv() -> bool:
    '''
    Are we in a virtual environment?
    '''
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix')
                                           and sys.base_prefix != sys.prefix)


def ensure_dependencies():
    '''
    Automatically install dependencies if they are not installed.
    '''
    p = Path(__file__).parent.parent / 'requirements.txt'
    if not p.exists():
        raise RuntimeError(f'Cannot find {p}')

    user = '' if is_in_venv() else '--user'

    with open(p) as f:
        deps = [i.strip().split('=')[0] for i in f.read().splitlines()]

    try:
        for dep in deps:
            importlib.import_module(dep)
    except ModuleNotFoundError:
        print('Installing dependencies...')
        if os.system(f'{sys.executable} -m pip install {user} -U pip'):
            raise Exception('Unable to upgrade pip!')
        if os.system(f'{sys.executable} -m pip install {user} -U -r {p}'):
            raise Exception('Unable to install dependencies!')
        os.execl(sys.executable, sys.executable, *sys.argv)


def chdir_to_root():
    '''
    Change working directory to the root of the repository
    '''
    root = Path('/')
    p = Path(__file__).resolve()
    while p != root:
        if (p / '.git').exists():
            os.chdir(p)
            break
        p = p.parent


def set_common_env():
    '''
    Set common environment variables.
    '''
    os.environ['TI_CI'] = '1'


_Environ = os.environ.__class__


class _EnvironWrapper(_Environ):
    def __setitem__(self, name: str, value: str) -> None:
        orig = self.get(name)
        _Environ.__setitem__(self, name, value)
        new = self[name]

        if orig == new:
            return

        from .escapes import escape_codes

        G = escape_codes['bold_green']
        R = escape_codes['bold_red']
        N = escape_codes['reset']
        print(f'{R}:: ENV -{name}={orig}{N}', file=sys.stderr, flush=True)
        print(f'{G}:: ENV +{name}={new}{N}', file=sys.stderr, flush=True)


def monkey_patch_environ():
    '''
    Monkey patch os.environ to print changes.
    '''
    os.environ.__class__ = _EnvironWrapper


def early_init():
    '''
    Do early initialization.
    This must be called before any other non-stdlib imports.
    '''
    ensure_dependencies()
    chdir_to_root()
    monkey_patch_environ()
    set_common_env()
