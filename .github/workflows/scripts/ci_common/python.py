# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from typing import Optional, Tuple
import sys

# -- third party --
# -- own --
from .misc import banner, env
from .tinysh import Command, sh


# -- code --
@banner('Setup Python {version}')
def setup_python(version: Optional[str] = None) -> Tuple[Command, Command]:
    '''
    Find the required Python environment and return the `python` and `pip` commands.
    '''
    assert version

    home = Path.home().resolve()

    for d in ['miniconda', 'miniconda3', 'miniforge3']:
        venv = home / d / 'envs' / version
        exe = venv / 'bin' / 'python'
        if not exe.exists():
            continue

        env['PATH'] = f'{venv / "bin"}:{env["PATH"]}'
        python = sh.bake(str(exe))
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
