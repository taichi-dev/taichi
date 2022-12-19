# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
from typing import Optional, Tuple
import os
import sys

# -- third party --
# -- own --
from .misc import banner
from .tinysh import Command, sh


# -- code --
@banner('Setup Python {version}')
def setup_python(version: Optional[str] = None) -> Tuple[Command, Command]:
    assert version

    home = Path.home().resolve()

    for d in ['miniconda', 'miniconda3', 'miniforge3']:
        env = home / d / 'envs' / version
        exe = env / 'bin' / 'python'
        if not exe.exists():
            continue

        os.environ['PATH'] = f'{env / "bin"}:{os.environ["PATH"]}'
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
