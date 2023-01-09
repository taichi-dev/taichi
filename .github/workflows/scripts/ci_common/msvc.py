#!/usr/bin/env python3

# -- stdlib --
import glob
import os
import platform

import setuptools.msvc
# -- third party --
# -- own --
from ci_common.dep import download_dep
from ci_common.python import setup_python
from ci_common.sccache import setup_sccache
from ci_common.tinysh import Command, environ, git, sh

from .misc import banner, hook


# -- code --
@hook(setuptools.msvc)
def isfile(orig, path):
    if path is None:
        return False
    return orig(path)


@banner('Setup VS2022')
def setup_vs2022(env_out: dict) -> None:
    '''
    Setup VS2022.
    '''
    import setuptools.msvc as msvc

    env = {k.upper(): v for k, v in os.environ.items()}
    vcvars = msvc.msvc14_get_vc_env('x64')

    v1 = {}
    for k, v in vcvars.items():
        if env.get(k.upper()) == v:
            continue
        v1[k.upper()] = v

    for k, v in v1.items():
        print(k, v)
