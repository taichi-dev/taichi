from .impl import *
from .matrix import Matrix
from .transformer import TaichiSyntaxError

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
  get_runtime().materialize()
  assert loss.snode().ptr.has_grad(), "gradient for loss not allocated"
  if clear_gradients:
    clear_all_gradients()
  loss[None] = 0
  loss.grad[None] = 1
  return runtime.get_tape(loss)

def clear_all_gradients():
  get_runtime().materialize()
  core.get_current_program().clear_all_gradients()

schedules = [parallelize, vectorize, block_dim, cache]
lang_core = core

def static_print(*args, __p=print, **kwargs):
  __p(*args, **kwargs)

# test x86_64 only
def simple_test(func):
  def test(*args, **kwargs):
    reset()
    cfg.arch = x86_64
    func(*args, **kwargs)

  return test


# test with all archs
def all_archs(func):
  import taichi as ti
  def test(*args, **kwargs):
    archs = [x86_64]
    if ti.core.with_cuda():
      archs.append(cuda)
    for arch in archs:
      reset()
      cfg.arch = arch
      func(*args, **kwargs)

  return test

# test with host arch only
def host_arch(func):
  import taichi as ti
  def test(*args, **kwargs):
    archs = [x86_64]
    for arch in archs:
      reset()
      cfg.arch = arch
      func(*args, **kwargs)

  return test

def must_throw(ex):
  def decorator(func):
    def func__(*args, **kwargs):
      finishes = False
      try:
        simple_test(func)(*args, **kwargs)
        finishes = True
      except ex:
        # throws. test passed
        pass
      except Exception as err_actual:
        assert False, 'Exception {} instead of {} thrown'.format(str(type(err_actual)), str(ex))
      if finishes:
        assert False, 'Test finishes instead of throwing {}'.format(str(ex))


    return func__
  return decorator


def complex_kernel(func):
  def decorated(*args, **kwargs):
    get_runtime().inside_complex_kernel = True
    if get_runtime().target_tape:
      get_runtime().target_tape.insert(decorated, args)
    try:
      with FrameBacktraceGuard(2):
        func(*args, **kwargs)
    except Exception as e:
      raise e
    finally:
      get_runtime().inside_complex_kernel = False
  decorated.grad = None
  return decorated

def complex_kernel_grad(primal):
  def decorator(func):
    def decorated(*args, **kwargs):
      with FrameBacktraceGuard(2):
        func(*args, **kwargs)
    primal.grad = decorated
    return decorated
  return decorator

def print(*args, **kwargs):
  import taichi as ti
  ti.error('ti.print is deprecated. Please simply use the builtin "print" in Python.')


__all__ = [s for s in dir() if not s.startswith('_')]
