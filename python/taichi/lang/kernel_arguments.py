from .core import taichi_lang_core
from .expr import Expr
import numpy as np
from .util import *

class ArgExtArray:
  def __init__(self, dim=1):
    assert dim == 1


ext_arr = ArgExtArray


class Template:
  def __init__(self, tensor=None, dim=None):
    self.tensor = tensor
    self.dim = dim


template = Template


def decl_arg(dt):
  if isinstance(dt, template):
    return
  if dt is np.ndarray or isinstance(dt, ext_arr):
    print("Warning: numpy array arg supports 1D and f32 only for now")
    id = taichi_lang_core.decl_arg(f32, True)
    return Expr(taichi_lang_core.make_external_tensor_expr(f32, 1, id))
  else:
    id = taichi_lang_core.decl_arg(dt, False)
    return Expr(taichi_lang_core.make_arg_load_expr(id))

