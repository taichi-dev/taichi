# -*- coding: utf-8 -*-

import platform
import shutil
from typing import Optional, Tuple

from .dep import download_dep
from .misc import banner, concat_paths, get_cache_home
from .tinysh import Command, sh


def setup_miniforge3(prefix):
    u = platform.uname()
    if u.system == "Linux":
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-Linux-x86_64.sh'
        download_dep(url, prefix, args=['-bfp', str(prefix)])
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-MacOSX-arm64.sh'
        download_dep(url, prefix, args=['-bfp', str(prefix)])
    elif (u.system, u.machine) == ("Darwin", "x86_64"):
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-MacOSX-x86_64.sh'
        download_dep(url, prefix, args=['-bfp', str(prefix)])
    elif u.system == "Windows":
        url = 'https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-Windows-x86_64.exe'
        download_dep(url,
                     prefix,
                     args=[
                         '/S',
                         '/InstallationType=JustMe',
                         '/RegisterPython=0',
                         '/KeepPkgCache=0',
                         '/AddToPath=0',
                         '/NoRegistry=1',
                         '/NoShortcut=1',
                         '/NoScripts=1',
                         '/CheckPathLength=1',
                         f'/D={prefix}',
                     ])
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")


@banner('Setup Python {version}')
def setup_python(env_out: dict,
                 version: Optional[str] = None) -> Tuple[Command, Command]:
    '''
    Find the required Python environment and return the `python` and `pip` commands.
    '''
    assert version

    windows = platform.system() == "Windows"

    prefix = get_cache_home() / 'miniforge3'
    setup_miniforge3(prefix)

    if windows:
        conda_path = prefix / 'Scripts' / 'conda.exe'
    else:
        conda_path = prefix / 'bin' / 'conda'

    if not conda_path.exists():
        shutil.rmtree(prefix, ignore_errors=True)
        setup_miniforge3(prefix)
        if not conda_path.exists():
            raise RuntimeError(f"Failed to setup miniforge3 at {prefix}")

    conda = sh.bake(str(conda_path))

    env = prefix / 'envs' / version
    if windows:
        exe = env / 'python.exe'
        env_out['PATH'] = concat_paths(env, env / 'Scripts',
                                       prefix / 'Library' / 'bin',
                                       env_out.get('PATH'))
        import os
        os.environ['PATH'] = concat_paths(env, env / 'Scripts',
                                          prefix / 'Library' / 'bin',
                                          os.environ.get('PATH'))
    else:
        exe = env / 'bin' / 'python'
        env_out['PATH'] = concat_paths(env / 'bin', env_out.get('PATH'))

    if not exe.exists():
        conda.create('-y', '-n', version, f'python={version}')

    python = sh.bake(str(exe))
    pip = python.bake('-m', 'pip')

    pip.install('-U', 'pip')
    pip.uninstall('-y', 'taichi', 'taichi-nightly')

    return python, pip
