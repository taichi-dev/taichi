from .core import taichi_lang_core

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

