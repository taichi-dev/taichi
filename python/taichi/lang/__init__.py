from .impl import *
from .matrix import Matrix
from .transformer import TaichiSyntaxError
from .ndrange import ndrange, GroupedNDRange

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
cross = Matrix.cross
dot = Matrix.dot
normalized = Matrix.normalized

cfg = default_cfg()
current_cfg = current_cfg()
x86_64 = core.x86_64
cuda = core.cuda
metal = core.metal
profiler_print = lambda: core.get_current_program().profiler_print()
profiler_clear = lambda: core.get_current_program().profiler_clear()


def reset():
  from .impl import reset as impl_reset
  impl_reset()
  global runtime
  runtime = get_runtime()

def init(default_fp=None, default_ip=None, print_processed=None, debug=None, **kwargs):
  if debug is None:
    import os
    debug = bool(int(os.environ.get('TI_DEBUG', '0')))

  import taichi as ti
  ti.reset()
  if default_fp is not None:
    ti.get_runtime().set_default_fp(default_fp)
  if default_ip is not None:
    ti.get_runtime().set_default_ip(default_ip)
  if print_processed is not None:
    ti.get_runtime().print_preprocessed = print_processed
  if debug:
    ti.set_logging_level(ti.DEBUG)
  ti.cfg.debug = debug
  for k, v in kwargs.items():
    setattr(ti.cfg, k, v)
  ti.get_runtime().create_program()

def cache_shared(v):
  taichi_lang_core.cache(0, v.ptr)


def cache_l1(v):
  taichi_lang_core.cache(1, v.ptr)


parallelize = core.parallelize
serialize = lambda: parallelize(1)
vectorize = core.vectorize
block_dim = core.block_dim
cache = core.cache

def inversed(x):
  return x.inversed()

transposed = Matrix.transposed

def polar_decompose(A, dt=None):
  if dt is None:
    dt = get_runtime().default_fp
  from .linalg import polar_decompose
  return polar_decompose(A, dt)

def svd(A, dt=None):
  if dt is None:
    dt = get_runtime().default_fp
  from .linalg import svd
  return svd(A, dt)

determinant = Matrix.determinant
trace = Matrix.trace


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
  
  import taichi as ti
  def visit(node):
    places = []
    for i in range(node.ptr.get_num_ch()):
      ch = node.ptr.get_ch(i)
      if not ch.is_place():
        visit(SNode(ch))
      else:
        if not ch.is_primal():
          places.append(ch.get_expr())
      
    places = tuple(places)
    if places:
      from .meta import clear_gradients
      clear_gradients(places)
    
  visit(ti.root)


schedules = [parallelize, vectorize, block_dim, cache]
lang_core = core


def static_print(*args, __p=print, **kwargs):
  __p(*args, **kwargs)

def benchmark(func, repeat=100, args=()):
  import taichi as ti
  import time
  for i in range(repeat // 3):
    func(*args) # compile the kernel first
  ti.sync()
  t = time.time()
  for n in range(repeat):
    func(*args)
  ti.get_runtime().sync()
  elapsed = time.time() - t
  return elapsed / repeat

# test x86_64 only
def simple_test(func):

  def test(*args, **kwargs):
    reset()
    cfg.arch = x86_64
    func(*args, **kwargs)

  return test

def supported_archs():
  import taichi as ti
  archs = [x86_64]
  if ti.core.with_cuda():
    archs.append(cuda)
  return archs
    
# test with all archs
def all_archs_with(**kwargs):
  def decorator(func):
    import taichi as ti
    def test(*func_args, **func_kwargs):
      for arch in ti.supported_archs():
        ti.init(arch=arch, **kwargs)
        func(*func_args, **func_kwargs)
  
    return test
  
  return decorator

# test with all archs
def all_archs(func):
  return all_archs_with()(func)


def torch_test(func):
  import taichi as ti
  if ti.has_pytorch():
    return ti.all_archs(func)
  else:
    return lambda: None

# test with host arch only
def host_arch(func):
  import taichi as ti

  def test(*args, **kwargs):
    archs = [x86_64]
    for arch in archs:
      ti.init(arch=arch)
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
        assert False, 'Exception {} instead of {} thrown'.format(
            str(type(err_actual)), str(ex))
      if finishes:
        assert False, 'Test successfully finished instead of throwing {}'.format(str(ex))

    return func__

  return decorator


def complex_kernel(func):

  def decorated(*args, **kwargs):
    get_runtime().inside_complex_kernel = True
    if get_runtime().target_tape:
      get_runtime().target_tape.insert(decorated, args)
    try:
      func(*args, **kwargs)
    finally:
      get_runtime().inside_complex_kernel = False

  decorated.grad = None
  return decorated


def complex_kernel_grad(primal):

  def decorator(func):

    def decorated(*args, **kwargs):
      func(*args, **kwargs)

    primal.grad = decorated
    return decorated

  return decorator

def sync():
  get_runtime().sync()

__all__ = [s for s in dir() if not s.startswith('_')]
