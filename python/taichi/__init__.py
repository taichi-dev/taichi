from taichi.main import main
from taichi.core import tc_core
from taichi.core import start_memory_monitoring, is_release, package_root
from taichi.misc.util import vec, veci, set_gdb_trigger, set_logging_level, info, warn, error, debug, trace, INFO, WARN, ERROR, DEBUG, TRACE
from taichi.core.util import require_version
from taichi.tools import *
from taichi.misc import *
from taichi.misc.gui import GUI
from taichi.misc.image import imread, imwrite, imshow
from taichi.misc.task import Task
from taichi.misc.test import approx
from taichi.misc import settings as settings
from taichi.misc.gui import rgb_to_hex
from taichi.misc.settings import *
from taichi.tools.video import VideoManager
from taichi.tools.file import *
from taichi.system import *
from taichi.lang import *
from .torch_io import from_torch, to_torch


def test():
    task = taichi.Task('test')
    return task.run([])


__all__ = [s for s in dir() if not s.startswith('_')] + ['settings']

__version__ = (core.get_version_major(), core.get_version_minor(),
               core.get_version_patch())
