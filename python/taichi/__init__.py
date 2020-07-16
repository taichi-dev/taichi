from .core import *
from .misc import *
from .lang import *  # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
from .tools import *
from .main import main
from .torch_io import from_torch, to_torch


def test():
    import taichi as ti
    task = ti.Task('test')
    return task.run([])


__all__ = [s for s in dir() if not s.startswith('_')]

__version__ = (core.get_version_major(), core.get_version_minor(),
               core.get_version_patch())
