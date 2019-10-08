import inspect
from .core import taichi_lang_core
from .transformer import ASTTransformer
from .expr import Expr
from .snode import SNode
import ast
import astor
from .util import *
import numpy as np

float32 = taichi_lang_core.DataType.float32
f32 = float32
float64 = taichi_lang_core.DataType.float64
f64 = float64

int32 = taichi_lang_core.DataType.int32
i32 = int32
int64 = taichi_lang_core.DataType.int64
i64 = int64


def decl_arg(dt):
  if dt is np.ndarray:
    print("Warning: numpy array arg supports 1D and f32 only for now")
    id = taichi_lang_core.decl_arg(f32, True)
    return Expr(taichi_lang_core.make_external_tensor_expr(f32, 1, id))
  else:
    id = taichi_lang_core.decl_arg(dt, False)
    return Expr(taichi_lang_core.make_arg_load_expr(id))

def expr_init(rhs):
  if rhs is None:
    return Expr(taichi_lang_core.expr_alloca())
  if is_taichi_class(rhs):
    return rhs.variable()
  else:
    if isinstance(rhs, list):
      return [expr_init(e) for e in rhs]
    elif isinstance(rhs, tuple):
      return tuple(expr_init(e) for e in rhs)
    else:
      return Expr(taichi_lang_core.expr_var(Expr(rhs).ptr))


def make_expr_group(*exprs):
  if len(exprs) == 1:
    from .matrix import Matrix
    if (isinstance(exprs[0], list) or isinstance(exprs[0], tuple)):
      exprs = exprs[0]
    elif isinstance(exprs[0], Matrix):
      mat = exprs[0]
      assert mat.m == 1
      exprs = mat.entries
  expr_group = taichi_lang_core.ExprGroup()
  for i in exprs:
    expr_group.push_back(Expr(i).ptr)
  return expr_group

def atomic_add(a, b):
  a.atomic_add(b)

def subscript(value, *indices):
  try:
    import numpy as np
    if isinstance(value, np.ndarray) or isinstance(value, list):
      return value.__getitem__(*indices)
  except:
    pass
  if isinstance(value, tuple) or isinstance(value, list):
    assert len(indices) == 1
    return value[indices[0]]
  if len(indices) == 1 and is_taichi_class(indices[0]):
    indices = indices[0].entries
  if is_taichi_class(value):
    return value.subscript(*indices)
  else:
    if isinstance(indices, tuple) and len(indices) == 1 and indices[0] is None:
      return Expr(
        taichi_lang_core.subscript(value.ptr, make_expr_group()))
    else:
      return Expr(
        taichi_lang_core.subscript(value.ptr, make_expr_group(*indices)))


class PyTaichi:
  def __init__(self):
    self.materialized = False
    self.prog = None
    self.layout_functions = []
    self.compiled_functions = {}
    self.compiled_grad_functions = {}
    self.scope_stack = []
    self.inside_kernel = False
    self.global_vars = []
    self.print_preprocessed = False
    self.default_fp = f32
    self.default_ip = i32
    self.target_tape = None
    Expr.materialize_layout_callback = self.materialize

  def set_default_fp(self, fp):
    assert fp in [f32, f64]
    self.default_fp = fp

  def set_default_ip(self, ip):
    assert ip in [i32, i64]
    self.default_ip = ip

  def materialize(self):
    assert self.materialized == False
    Expr.layout_materialized = True
    self.prog = taichi_lang_core.Program()

    def layout():
      for func in self.layout_functions:
        func()

    print("Materializing layout...".format())
    taichi_lang_core.layout(layout)
    self.materialized = True
    for var in self.global_vars:
      assert var.ptr.snode() is not None, 'variable not placed.'

  def clear(self):
    if self.prog:
      self.prog.finalize()
      self.prog = None
    Expr.materialize_layout_callback = None
    Expr.layout_materialized = False
    
  def get_tape(self, loss=None):
    from .tape import Tape
    return Tape(self, loss)
  
  def sync(self):
    self.prog.synchronize()


pytaichi = PyTaichi()

def make_constant_expr(val):
  if isinstance(val, int):
    if pytaichi.default_ip == i32:
      return Expr(taichi_lang_core.make_const_expr_i32(val))
    elif pytaichi.default_ip == i64:
      return Expr(taichi_lang_core.make_const_expr_i64(val))
    else:
      assert False
  else:
    if pytaichi.default_fp == f32:
      return Expr(taichi_lang_core.make_const_expr_f32(val))
    elif pytaichi.default_fp == f64:
      return Expr(taichi_lang_core.make_const_expr_f64(val))
    else:
      assert False

def reset():
  global pytaichi
  global root
  pytaichi.clear()
  pytaichi = PyTaichi()
  taichi_lang_core.reset_default_compile_config()
  root = SNode(taichi_lang_core.get_root())


def inside_kernel():
  return pytaichi.inside_kernel


def remove_indent(lines):
  lines = lines.split('\n')
  to_remove = 0
  for i in range(len(lines[0])):
    if lines[0][i] == ' ':
      to_remove = i + 1
    else:
      break

  cleaned = []
  for l in lines:
    cleaned.append(l[to_remove:])
    if len(l) >= to_remove:
      for i in range(to_remove):
        assert l[i] == ' '

  return '\n'.join(cleaned)

def func(foo):
  src = remove_indent(inspect.getsource(foo))
  tree = ast.parse(src)

  func_body = tree.body[0]
  func_body.decorator_list = []

  visitor = ASTTransformer(False)
  visitor.visit(tree)
  ast.fix_missing_locations(tree)

  if pytaichi.print_preprocessed:
    print(astor.to_source(tree.body[0], indent_with='  '))

  ast.increment_lineno(tree, inspect.getsourcelines(foo)[1] - 1)

  pytaichi.inside_kernel = True
  frame = inspect.currentframe().f_back
  exec(compile(tree, filename=inspect.getsourcefile(foo), mode='exec'),
       dict(frame.f_globals, **frame.f_locals), locals())
  pytaichi.inside_kernel = False
  compiled = locals()[foo.__name__]
  return compiled


class Kernel:
  def __init__(self, foo, is_grad):
    self.foo = foo
    self.is_grad = is_grad
    self.materialized = False
    if is_grad:
      self.compiled_functions = pytaichi.compiled_functions
    else:
      self.compiled_functions = pytaichi.compiled_grad_functions
  
  
  def materialize(self, extra_frame_backtrace=-1):
    if not self.materialized:
      self.materialized = True
    else:
      return
    if not pytaichi.materialized:
      pytaichi.materialize()
    grad_suffix = ""
    if self.is_grad:
      grad_suffix = ".grad"
    print("Compiling kernel {}{}...".format(self.foo.__name__, grad_suffix))
  
    src = remove_indent(inspect.getsource(self.foo))
    tree = ast.parse(src)
    # print(astor.to_source(tree.body[0]))
  
    func_body = tree.body[0]
    func_body.decorator_list = []
  
    visitor = ASTTransformer()
    visitor.visit(tree)
    ast.fix_missing_locations(tree)
  
    if pytaichi.print_preprocessed:
      print(astor.to_source(tree.body[0], indent_with='  '))
  
    ast.increment_lineno(tree, inspect.getsourcelines(self.foo)[1] - 1)
  
    pytaichi.inside_kernel = True
    frame = inspect.currentframe()
    for t in range(extra_frame_backtrace + 2):
      frame = frame.f_back
    exec(compile(tree, filename=inspect.getsourcefile(self.foo), mode='exec'),
         dict(frame.f_globals, **frame.f_locals), locals())
    pytaichi.inside_kernel = False
    compiled = locals()[self.foo.__name__]
  
    t_kernel = taichi_lang_core.create_kernel(self.foo.__name__, self.is_grad)
    t_kernel = t_kernel.define(lambda: compiled())
    
    # The actual function body
    def func__(*args):
      for i, v in enumerate(args):
        if isinstance(v, float):
          t_kernel.set_arg_float(i, v)
        elif isinstance(v, int):
          t_kernel.set_arg_int(i, v)
        elif isinstance(v, np.ndarray):
          tmp = np.ascontiguousarray(v)
          t_kernel.set_arg_nparray(i, int(tmp.ctypes.data), tmp.nbytes)
        else:
          assert False, 'Argument to kernels must have type float/int'
      if pytaichi.target_tape:
        pytaichi.target_tape.insert(self, args)
      t_kernel()
    self.compiled_functions[self.foo] = func__
    
  def __call__(self, *args, extra_frame_backtrace=0):
    self.materialize(extra_frame_backtrace=extra_frame_backtrace)
    self.compiled_functions[self.foo](*args)

def kernel(foo):
  ret = Kernel(foo, False)
  ret.grad = Kernel(foo, True)
  return ret


def global_var(dt):
  # primal
  x = Expr(taichi_lang_core.make_id_expr(""))
  x.ptr = taichi_lang_core.global_new(x.ptr, dt)
  x.ptr.set_is_primal(True)
  pytaichi.global_vars.append(x)

  if taichi_lang_core.needs_grad(dt):
    # adjoint
    x_grad = Expr(taichi_lang_core.make_id_expr(""))
    x_grad.ptr = taichi_lang_core.global_new(x_grad.ptr, dt)
    x_grad.ptr.set_is_primal(False)
    x.set_grad(x_grad)

  return x


var = global_var

root = SNode(taichi_lang_core.get_root())


def layout(func):
  assert not pytaichi.materialized, "All layout must be specified before the first kernel launch / data access."
  pytaichi.layout_functions.append(func)



def tprint(var):
  code = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
  arg_name = code[code.index('(') + 1: code.index(')')]
  taichi_lang_core.print_(Expr(var).ptr, arg_name)


def indices(*x):
  return [taichi_lang_core.Index(i) for i in x]


index = indices


def static(x):
  return x


def current_cfg():
  return taichi_lang_core.current_compile_config()


def default_cfg():
  return taichi_lang_core.default_compile_config()

unary_ops = []

def unary(x):
  unary_ops.append(x)
  return x

binary_ops = []
def binary(foo):
  def x_(a, b):
    return foo(Expr(a), Expr(b))
  binary_ops.append(x_)
  return x_

def pow(x, n):
  assert isinstance(n, int) and n >= 0
  if n == 0:
    return 1
  ret = x
  for i in range(n - 1):
    ret = ret * x
  return ret

def logical_and(a, b):
  return a.logical_and(b)


def logical_or(a, b):
  return a.logical_or(b)

def logical_not(a):
  return a.logical_not()


def cast(obj, type):
  if is_taichi_class(obj):
    return obj.cast(type)
  else:
    return Expr(taichi_lang_core.value_cast(Expr(obj).ptr, type))

def sqr(obj):
  return obj * obj

@unary
def sin(expr):
  return Expr(taichi_lang_core.expr_sin(expr.ptr))

@unary
def cos(expr):
  return Expr(taichi_lang_core.expr_cos(expr.ptr))

@unary
def sqrt(expr):
  return Expr(taichi_lang_core.expr_sqrt(expr.ptr))

@unary
def floor(expr):
  return Expr(taichi_lang_core.expr_floor(expr.ptr))

@unary
def inv(expr):
  return Expr(taichi_lang_core.expr_inv(expr.ptr))

@unary
def tan(expr):
  return Expr(taichi_lang_core.expr_tan(expr.ptr))

@unary
def tanh(expr):
  return Expr(taichi_lang_core.expr_tanh(expr.ptr))

@unary
def exp(expr):
  return Expr(taichi_lang_core.expr_exp(expr.ptr))

@unary
def log(expr):
  return Expr(taichi_lang_core.expr_log(expr.ptr))

@unary
def abs(expr):
  return Expr(taichi_lang_core.expr_abs(expr.ptr))

def random(dt=f32):
  return Expr(taichi_lang_core.make_rand_expr(dt))

@binary
def max(a, b):
  return Expr(taichi_lang_core.expr_max(a.ptr, b.ptr))

@binary
def min(a, b):
  return Expr(taichi_lang_core.expr_min(a.ptr, b.ptr))

def append(l, indices, val):
  taichi_lang_core.insert_append(l.ptr, make_expr_group(indices), val.ptr)

def length(l, indices):
  return taichi_lang_core.insert_len(l.ptr, make_expr_group(indices))

