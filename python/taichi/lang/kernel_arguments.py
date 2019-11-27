from .core import taichi_lang_core
from .expr import Expr
import numpy as np
from .util import *

class ArgExtArray:
  def __init__(self, dim=1):
    assert dim == 1

  def extract(self, x):
    return x.dtype, len(x.shape)


ext_arr = ArgExtArray


class Template:
  def __init__(self, tensor=None, dim=None):
    self.tensor = tensor
    self.dim = dim

  def extract(self, x):
    return x


template = Template


def decl_arg(ty, dt):
  if isinstance(dt, template):
    return
  if ty == 'array':
    print("Warning: external array arg supports 1D only")
    id = taichi_lang_core.decl_arg(dt, True)
    return Expr(taichi_lang_core.make_external_tensor_expr(dt, 1, id))
  elif ty == 'scalar':
    id = taichi_lang_core.decl_arg(dt, False)
    return Expr(taichi_lang_core.make_arg_load_expr(id))
  else:
    assert False

