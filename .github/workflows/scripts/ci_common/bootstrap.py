# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import importlib
import os
import sys

# -- third party --
# -- own --

# -- code --
def ensure_dependencies():
    p = Path(__file__).parent.parent / 'requirements.txt'
    if not p.exists():
        raise RuntimeError(f'Cannot find {p}')

    with open(p) as f:
        deps = [i.strip().split('=')[0] for i in f.read().splitlines()]

    try:
        for dep in deps:
            importlib.import_module(dep)
    except ModuleNotFoundError:
        print('Installing dependencies...')
        if os.system(f'{sys.executable} -m pip install --user -U -r {p}'):
            raise Exception('Unable to install dependencies!')
        os.execl(sys.executable, sys.executable, *sys.argv)


def chdir_to_root():
    # Change working directory to the root of the repository
    root = Path('/')
    p = Path(__file__).resolve()
    while p != root:
        if (p / '.git').exists():
            os.chdir(p)
            break
        p = p.parent


def set_common_env():
    os.environ['TI_CI'] = '1'


def early_init():
    ensure_dependencies()
    chdir_to_root()
    set_common_env()
