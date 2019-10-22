from .impl import *
from .matrix import Matrix

core = taichi_lang_core
runtime = get_runtime()

i = indices(0)
j = indices(1)
k = indices(2)
l = indices(3)
ij = indices(0, 1)
ijk = indices(0, 1, 2)
ijkl = indices(0, 1, 2, 3)
Vector = Matrix
outer_product = Matrix.outer_product
cfg = default_cfg()
current_cfg = current_cfg()
x86_64 = core.x86_64
cuda = core.gpu
profiler_print = lambda: core.get_current_program().profiler_print()
profiler_clear = lambda: core.get_current_program().profiler_clear()

def reset():
  from .impl import reset as impl_reset
  impl_reset()
  global runtime
  runtime = get_runtime()

def cache_shared(v):
  taichi_lang_core.cache(0, v.ptr)

def cache_l1(v):
  taichi_lang_core.cache(1, v.ptr)

parallelize = core.parallelize
vectorize = core.vectorize
block_dim = core.block_dim
cache = core.cache
transposed = Matrix.transposed
polar_decompose = Matrix.polar_decompose
determinant = Matrix.determinant
set_default_fp = pytaichi.set_default_fp

def Tape(loss, clear_gradients=True):
  if clear_gradients:
    clear_all_gradients()
  loss[None] = 0
  loss.grad[None] = 1
  return runtime.get_tape(loss)

def clear_all_gradients():
  core.get_current_program().clear_all_gradients()

schedules = [parallelize, vectorize, block_dim, cache]
lang_core = core

print = tprint

def program_test(func):
  def test(*args, **kwargs):
    for arch in [x86_64, cuda]:
      reset()
      cfg.arch = arch
      func(*args, **kwargs)

  return test


__all__ = [s for s in dir() if not s.startswith('_')]

