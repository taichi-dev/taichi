import inspect
from core import taichi_lang
import ast
import astpretty
import astor

def expr_init(rhs):
  return Expr(taichi_lang.expr_var(Expr(rhs).ptr))

def expr_assign(lhs, rhs):
  taichi_lang.expr_assign(lhs.ptr, Expr(rhs).ptr)


class Expr:
  def __init__(self, *args):
    if len(args) == 1:
      if isinstance(args[0], taichi_lang.Expr):
        self.ptr = args[0]
      elif isinstance(args[0], Expr):
        self.ptr = args[0].ptr
      else:
        self.ptr = taichi_lang.make_constant_expr(args[0])
    else:
      assert False

  def __add__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_add(self.ptr, other.ptr))

  __radd__ = __add__

  def __sub__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_sub(self.ptr, other.ptr))

  def __mul__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_mul(self.ptr, other.ptr))

  __rmul__ = __mul__

  def __div__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_div(self.ptr, other.ptr))


  def __le__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_le(self.ptr, other.ptr))

  __rle__ = __le__

  def __lt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_lt(self.ptr, other.ptr))

  def __ge__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_ge(self.ptr, other.ptr))

  def __gt__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_gt(self.ptr, other.ptr))

  def __eq__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_eq(self.ptr, other.ptr))

  def __ne__(self, other):
    other = Expr(other)
    return Expr(taichi_lang.expr_cmp_ne(self.ptr, other.ptr))

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
        value=ast.Name(id='taichi_lang', ctx=ast.Load()), attr='expr_init',
        ctx=ast.Load())
    '''
    rhs = ast.Call(
      func=ast.Name(id='expr_init', ctx=ast.Load()),
      args=[node.value],
      keywords=[],
    )
    return ast.copy_location(ast.Assign(targets=node.targets, value=rhs), node)

compiled_functions = {}

def comp(foo):
  def ret():
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

      kernel = taichi_lang.create_kernel("test")
      kernel = kernel.define(lambda: compiled())
      compiled_functions[foo] = lambda: kernel()
    compiled_functions[foo]()
  return ret

x = Expr(taichi_lang.make_id_expr(""))


def tiprint(var):
  taichi_lang.print_(Expr(var).ptr, 'var')

@comp
def test():
  a = 1
  b = 2
  c = a + b * b
  tiprint(c)
  tiprint(1)
  tiprint(1 + 11)
  tiprint(x)


if __name__ == '__main__':
  prog = taichi_lang.Program()
  x.ptr = taichi_lang.global_new(x.ptr, taichi_lang.DataType.float32)
  def l():
    taichi_lang.get_root().place(x.ptr)
  taichi_lang.layout(l)
  test()
