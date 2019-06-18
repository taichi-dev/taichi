import inspect
from .core import taichi_lang_core
from .transformer import ASTTransformer
from .expr import Expr
from .snode import SNode
import ast
import astpretty
import astor
from .util import *

float32 = taichi_lang_core.DataType.float32
f32 = float32
float64 = taichi_lang_core.DataType.float64
f64 = float64

int32 = taichi_lang_core.DataType.int32
i32 = int32
int64 = taichi_lang_core.DataType.int64
i64 = int64


def decl_arg(dt):
  id = taichi_lang_core.decl_arg(dt)
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
  expr_group = taichi_lang_core.ExprGroup()
  for i in exprs:
    expr_group.push_back(Expr(i).ptr)
  return expr_group

def atomic_add(a, b):
  a.atomic_add(b)

def subscript(value, *indices):
  if len(indices) == 1 and is_taichi_class(indices[0]):
    indices = indices[0].entries
  if isinstance(value, tuple) or isinstance(value, list):
    assert len(indices) == 1
    return value[indices[0]]
  elif is_taichi_class(value):
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

    taichi_lang_core.layout(layout)
    self.materialized = True
    for var in self.global_vars:
      assert var.ptr.snode() is not None, 'variable not placed.'

  def clear(self):
    del self.prog
    Expr.materialize_layout_callback = None
    Expr.layout_materialized = False


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


def kernel(foo):
  def invoke(grad=False, *args, **kwargs):
    def ret(*args):
      if grad:
        compiled_functions = pytaichi.compiled_functions
      else:
        compiled_functions = pytaichi.compiled_grad_functions
      if not pytaichi.materialized:
        pytaichi.materialize()
      if foo not in compiled_functions:
        src = remove_indent(inspect.getsource(foo))
        tree = ast.parse(src)
        # print(astor.to_source(tree.body[0]))

        func_body = tree.body[0]
        func_body.decorator_list = []

        visitor = ASTTransformer()
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

        t_kernel = taichi_lang_core.create_kernel(foo.__name__, grad)
        t_kernel = t_kernel.define(lambda: compiled())
        def func__(*args):
          for i, v in enumerate(args):
            if isinstance(v, float):
              t_kernel.set_arg_float(i, v)
            elif isinstance(v, int):
              t_kernel.set_arg_int(i, v)
            else:
              assert False, 'Argument to kernels must have type float/int'
          t_kernel()
        compiled_functions[foo] = func__
      compiled_functions[foo](*args)
    return ret

  ret = invoke(False)
  ret.grad = invoke(True)
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


def cast(obj, type):
  if is_taichi_class(obj):
    return obj.cast(type)
  else:
    return Expr(taichi_lang_core.value_cast(Expr(obj).ptr, type))


def sqr(obj):
  return obj * obj


def static(x):
  return x


def current_cfg():
  return taichi_lang_core.current_compile_config()


def default_cfg():
  return taichi_lang_core.default_compile_config()


def logical_and(a, b):
  return a.logical_and(b)


def logical_or(a, b):
  return a.logical_or(b)


unary_ops = []


def unary(x):
  unary_ops.append(x)
  return x

def pow(x, n):
  assert isinstance(n, int) and n >= 0
  if n == 0:
    return 1
  ret = x
  for i in range(n - 1):
    ret = ret * x
  return ret


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
  return Expr(taichi_lang_core.expr_sqrt(expr.ptr))

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

def random(dt=f32):
  return Expr(taichi_lang_core.make_rand_expr(dt))
