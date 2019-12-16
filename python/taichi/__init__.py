from taichi.main import main
from taichi.core import tc_core
from taichi.core import start_memory_monitoring, is_release, package_root
from taichi.misc.util import vec, veci, set_gdb_trigger, set_logging_level, info, warning, error, debug, trace
from taichi.tools import *
from taichi.misc import *
from taichi.misc.task import Task
from taichi.misc import settings as settings
from taichi.misc.settings import *
from taichi.tools.video import VideoManager
from taichi.tools.file import *
from taichi.system import *
from taichi.lang import *
from .torch_io import from_torch, to_torch

GUI = core.GUI


def set_image(self, img):
  import numpy as np
  import taichi as ti
  if isinstance(img, ti.Matrix):
    img = img.to_numpy(as_vector=True)
  if isinstance(img, ti.Expr):
    img = img.to_numpy()
  assert isinstance(img, np.ndarray)
  assert len(img.shape) in [2, 3]
  img = img.astype(np.float32)
  if len(img.shape) == 2:
    img = img[..., None]
  if img.shape[2] == 1:
    img = img + np.zeros(shape=(1, 1, 4))
  if img.shape[2] == 3:
    img = np.concatenate([
        img,
        np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.float32)
    ],
                         axis=2)
  img = img.astype(np.float32)
  self.set_img(np.ascontiguousarray(img).ctypes.data)


GUI.set_image = set_image


def test():
  task = taichi.Task('test')
  return task.run([])


__all__ = [s for s in dir() if not s.startswith('_')] + ['settings']
