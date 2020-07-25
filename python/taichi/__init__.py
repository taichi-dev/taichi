from .core import *
from .misc import *
from .lang import *  # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
from .tools import *
from .main import main
from .torch_io import from_torch, to_torch
from .testing import *

__all__ = ['core', 'misc', 'lang', 'tools', 'main', 'torch_io']

__version__ = (core.get_version_major(), core.get_version_minor(),
               core.get_version_patch())
