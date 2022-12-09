# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import os
import sys

# -- third party --
# -- own --
from .tinysh import sh
from .misc import banner


# -- code --
@banner('Setup Python {version}')
def setup_python(version=None):
    assert version

    home = Path.home()

    for d in ['miniconda', 'miniconda3', 'miniforge3']:
        p = home / d / 'envs' / version / 'bin' / 'python'
        if not p.exists():
            continue

        python = sh.bake(str(p))
        pip = python.bake('-m', 'pip')
        break
    else:
        v = sys.version_info
        if f'{v.major}.{v.minor}' == version:
            python = sh.bake(sys.executable)
            pip = python.bake('-m', 'pip')
        else:
            raise ValueError(f'No python {version} found')

    pip.install('-U', 'pip')
    pip.uninstall('-y', 'taichi', 'taichi-nightly')

    return python, pip
