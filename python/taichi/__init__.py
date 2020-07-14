from taichi.misc.timing import profiler

profiler('main')
from taichi.main import main
profiler('core')
from taichi.core import ti_core
from taichi.core import start_memory_monitoring, is_release, package_root
profiler('misc.util')
from taichi.misc.util import vec, veci, set_gdb_trigger, print_profile_info, set_logging_level, info, warn, error, debug, trace, INFO, WARN, ERROR, DEBUG, TRACE
from taichi.core.util import require_version
profiler('tools')
from taichi.tools import *
profiler('misc.*')
from taichi.misc import *
from taichi.misc.gui import GUI
from taichi.misc.np2ply import PLYWriter
from taichi.misc.image import imread, imwrite, imshow, imdisplay
from taichi.misc.task import Task
from taichi.misc.test import *
from taichi.misc import settings as settings
from taichi.misc.gui import rgb_to_hex
from taichi.misc.error import *
from taichi.misc.settings import *
profiler('tools.{video,file}')
from taichi.tools.video import VideoManager
from taichi.tools.file import *
profiler('lang')
from taichi.lang import *
profiler('torch_io')
from .torch_io import from_torch, to_torch


def test():
    task = taichi.Task('test')
    return task.run([])


__all__ = [s for s in dir() if not s.startswith('_')] + ['settings']

__version__ = (core.get_version_major(), core.get_version_minor(),
               core.get_version_patch())
profiler()
