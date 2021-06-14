import sys
if sys.version_info[0] == 3:
    # Abort importing Taichi in a Python2 environment. 
    # This is to make Houdini load correctly under a Taichi dev install.
    from .core import *
    from .lang import *  # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
    from .main import main
    from .misc import *
    from .testing import *
    from .tools import *
    from .torch_io import from_torch, to_torch

    __all__ = ['core', 'misc', 'lang', 'tools', 'main', 'torch_io']

    __version__ = (core.get_version_major(), core.get_version_minor(),
                core.get_version_patch())
