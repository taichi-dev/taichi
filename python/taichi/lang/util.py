from .core import taichi_lang_core
import numpy as np

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
