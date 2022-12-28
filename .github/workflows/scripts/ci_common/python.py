# -*- coding: utf-8 -*-

import os
import platform
import shutil
from typing import Optional, Tuple

from .dep import download_dep
from .misc import banner, get_cache_home
from .tinysh import Command, sh


def setup_miniforge3(prefix):
    u = platform.uname()
    if u.system == "Linux":
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-Linux-x86_64.sh'
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-MacOSX-arm64.sh'
    elif (u.system, u.machine) == ("Darwin", "x86_64"):
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-MacOSX-x86_64.sh'
    elif u.system == "Windows":
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-Windows-x86_64.exe'
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    download_dep(url, prefix, args=['-bfp', str(prefix)])


@banner('Setup Python {version}')
def setup_python(env_out: dict,
                 version: Optional[str] = None) -> Tuple[Command, Command]:
    '''
    Find the required Python environment and return the `python` and `pip` commands.
    '''
    assert version

    prefix = get_cache_home() / 'miniforge3'
    setup_miniforge3(prefix)
    conda_path = prefix / 'bin' / 'conda'
    if not conda_path.exists():
        shutil.rmtree(prefix, ignore_errors=True)
        setup_miniforge3(prefix)
        if not conda_path.exists():
            raise RuntimeError(f"Failed to setup miniforge3 at {prefix}")

    conda = sh.bake(str(conda_path))

    env = prefix / 'envs' / version
    exe = env / 'bin' / 'python'

    if not exe.exists():
        conda.create('-y', '-n', version, f'python={version}')

    env_out['PATH'] = f'{env / "bin"}:{env_out["PATH"]}'
    python = sh.bake(str(exe))
    pip = python.bake('-m', 'pip')

    pip.install('-U', 'pip')
    pip.uninstall('-y', 'taichi', 'taichi-nightly')

    return python, pip
