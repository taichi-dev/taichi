import inspect
from core import taichi_lang_core
from .transformer import ASTTransformer
from .expr import Expr
import ast
import astpretty
import astor

def expr_init(rhs):
  return Expr(taichi_lang_core.expr_var(Expr(rhs).ptr))


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
      print(astor.to_source(tree.body[0]))

      ast.increment_lineno(tree, inspect.getsourcelines(foo)[1] - 1)

      exec(compile(tree, filename=inspect.getsourcefile(foo), mode='exec'), inspect.currentframe().f_back.f_globals, locals())
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

