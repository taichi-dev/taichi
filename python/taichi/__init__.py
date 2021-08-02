from taichi.core import *
from taichi.lang import *  # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
from taichi.main import main
from taichi.misc import *
from taichi.testing import *
from taichi.tools import *
from taichi.torch_io import from_torch, to_torch

# Issue#2223: Do not reorder, or we're busted with partially initialized module
from taichi import aot  # isort:skip

__all__ = ['core', 'misc', 'lang', 'tools', 'main', 'torch_io']

__version__ = (core.get_version_major(), core.get_version_minor(),
               core.get_version_patch())
