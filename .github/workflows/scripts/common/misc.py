# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import platform

# -- third party --
# -- own --

# -- code --
def is_manylinux2014():
    # manylinux2014 builds in a special CentOS docker image
    return platform.system() == 'Linux' and Path('/etc/centos-release').exists()
