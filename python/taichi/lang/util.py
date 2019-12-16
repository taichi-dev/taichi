from .core import taichi_lang_core
import numpy as np

_has_pytorch = False

try:
  import torch
  _has_pytorch = True
except:
  pass


def has_pytorch():
  return _has_pytorch


def is_taichi_class(rhs):
  taichi_class = False
  try:
    if rhs.is_taichi_class:
      taichi_class = True
  except:
    pass
  return taichi_class


float32 = taichi_lang_core.DataType.float32
f32 = float32
float64 = taichi_lang_core.DataType.float64
f64 = float64

int32 = taichi_lang_core.DataType.int32
i32 = int32
int64 = taichi_lang_core.DataType.int64
i64 = int64


def to_numpy_type(dt):
  if dt == f32:
    return np.float32
  elif dt == f64:
    return np.float64
  elif dt == i32:
    return np.int32
  elif dt == i64:
    return np.int64
  else:
    assert False


def to_pytorch_type(dt):
  import torch
  if dt == f32:
    return torch.float32
  elif dt == f64:
    return torch.float64
  elif dt == i32:
    return torch.int32
  elif dt == i64:
    return torch.int64
  else:
    assert False


def to_taichi_type(dt):
  if type(dt) == taichi_lang_core.DataType:
    return dt
  if dt == np.float32:
    return f32
  elif dt == np.float64:
    return f64
  elif dt == np.int32:
    return i32
  elif dt == np.int64:
    return i64

  if has_pytorch():
    if dt == torch.float32:
      return f32
    elif dt == torch.float64:
      return f64
    elif dt == torch.int32:
      return i32
    elif dt == torch.int64:
      return i64

  raise AssertionError("Unknown type {}".format(dt))
