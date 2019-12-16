from .core import taichi_lang_core
from .expr import Expr
import numpy as np
from .util import *


class ArgExtArray:

  def __init__(self, dim=1):
    assert dim == 1

  def extract(self, x):
    return to_taichi_type(x.dtype), len(x.shape)


ext_arr = ArgExtArray


class Template:

  def __init__(self, tensor=None, dim=None):
    self.tensor = tensor
    self.dim = dim

  def extract(self, x):
    return x


template = Template


def decl_scalar_arg(dt):
  id = taichi_lang_core.decl_arg(dt, False)
  return Expr(taichi_lang_core.make_arg_load_expr(id))


def decl_ext_arr_arg(dt, dim):
  id = taichi_lang_core.decl_arg(dt, True)
  return Expr(taichi_lang_core.make_external_tensor_expr(dt, dim, id))
