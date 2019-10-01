from taichi.main import main
from taichi.core import tc_core
from taichi.core import start_memory_monitoring
from taichi.misc.util import Vector, Vectori, set_gdb_trigger, set_logging_level
from taichi.tools import *
from taichi.misc import *
from taichi.misc.task import Task
from taichi.misc import settings as settings
from taichi.misc.settings import *
from taichi.tools.video import VideoManager
from taichi.tools.file import *
from taichi.system import *
from taichi.lang import *

def test():
  task = taichi.Task('test')
  return task.run([])

__all__ = [s for s in dir() if not s.startswith('_')] + ['settings']
