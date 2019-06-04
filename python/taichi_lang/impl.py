import inspect
from core import taichi_lang_core
import ast
import astpretty
import astor

def expr_init(rhs):
  return Expr(taichi_lang_core.expr_var(Expr(rhs).ptr))

def expr_assign(lhs, rhs):
  taichi_lang_core.expr_assign(lhs.ptr, Expr(rhs).ptr)

class PyTaichi:
  def __init__(self):
    self.materialized = False
    self.prog = None
    self.layout_functions = []
    self.compiled_functions = {}


  def materialize(self):
    assert self.materialized == False
    self.prog = taichi_lang_core.Program()
    def layout():
      for func in self.layout_functions:
        func()
    taichi_lang_core.layout(layout)
    self.materialized = True

pytaichi = PyTaichi()


class Expr:
  def __init__(self, *args):
    if len(args) == 1:
      if isinstance(args[0], taichi_lang_core.Expr):
        self.ptr = args[0]
      elif isinstance(args[0], Expr):
        self.ptr = args[0].ptr
      else:
        self.ptr = taichi_lang_core.make_constant_expr(args[0])
    else:
      assert False

  def __add__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_add(self.ptr, other.ptr))

  __radd__ = __add__

  def __sub__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_sub(self.ptr, other.ptr))

  def __mul__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_mul(self.ptr, other.ptr))

  __rmul__ = __mul__

  def __div__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_div(self.ptr, other.ptr))


  def __le__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_le(self.ptr, other.ptr))

  __rle__ = __le__

  def __lt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_lt(self.ptr, other.ptr))

  def __ge__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_ge(self.ptr, other.ptr))

  def __gt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_gt(self.ptr, other.ptr))

  def __eq__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_eq(self.ptr, other.ptr))

  def __ne__(self, other):
    other = Expr(other)
    return Expr(taichi_lang_core.expr_cmp_ne(self.ptr, other.ptr))

  def __getitem__(self, item):
    item = Expr(item)
    return Expr(expr_index(self, item.ptr))

  def serialize(self):
    return self.ptr.serialize()


class ASTTransformer(ast.NodeTransformer):
  def visit_Assign(self, node):
    assert (len(node.targets) == 1)
    '''
    ast.Attribute(
        value=ast.Name(id='taichi_lang_core', ctx=ast.Load()), attr='expr_init',
        ctx=ast.Load())
    '''
    rhs = ast.Call(
      func=ast.Name(id='expr_init', ctx=ast.Load()),
      args=[node.value],
      keywords=[],
    )
    return ast.copy_location(ast.Assign(targets=node.targets, value=rhs), node)

def kernel(foo):
  def ret():
    compiled_functions = pytaichi.compiled_functions
    if not pytaichi.materialized:
      pytaichi.materialize()
    if foo not in compiled_functions:
      src = inspect.getsource(foo)
      tree = ast.parse(src)
      # print(astor.to_source(tree.body[0]))

      func_body = tree.body[0]
      func_body.decorator_list = []

      visitor = ASTTransformer()
      visitor.visit(tree)
      ast.fix_missing_locations(tree)

      # astpretty.pprint(func_body)
      # print(codegen.to_source(tree))
      print(astor.to_source(tree.body[0]))

      exec(compile(tree, filename='tmp', mode='exec'))
      compiled = locals()[foo.__name__]

      t_kernel = taichi_lang_core.create_kernel("test")
      t_kernel = t_kernel.define(lambda: compiled())
      compiled_functions[foo] = lambda: t_kernel()
    compiled_functions[foo]()
  return ret

def global_var(dt):
  x = Expr(taichi_lang_core.make_id_expr(""))
  x.ptr = taichi_lang_core.global_new(x.ptr, dt)
  return x

root = taichi_lang_core.get_root()

def layout(func):
  assert not pytaichi.materialized, "All layout must be specified before the first kernel launch."
  pytaichi.layout_functions.append(func)

float32 = taichi_lang_core.DataType.float32
f32 = float32

def tprint(var):
  taichi_lang_core.print_(Expr(var).ptr, 'var')

