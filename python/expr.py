import inspect
from core import taichi_lang
import ast
import astpretty
import astor

def expr_init(rhs):
  return taichi_lang.expr_var(rhs.ptr)

def expr_assign(lhs, rhs):
  taichi_lang.expr_assign(lhs.ptr, rhs.ptr)


class Expr:
  def __init__(self, *args):
    if len(args) == 1:
      if isinstance(args[0], taichi_lang.Expr):
        self.ptr = args[0]
      else:
        self.ptr = taichi_lang.make_constant_expr(args[0])
    else:
      assert False

  def __add__(self, other):
    return Expr(taichi_lang.expr_add(self.ptr, other.ptr))

  def __sub__(self, other):
    return Expr(taichi_lang.expr_sub(self.ptr, other.ptr))

  def __mul__(self, other):
    return Expr(taichi_lang.expr_mul(self.ptr, other.ptr))

  def __div__(self, other):
    return Expr(taichi_lang.expr_div(self.ptr, other.ptr))

  def __le__(self, other):
    return Expr(taichi_lang.expr_cmp_le(self.ptr, other.ptr))

  def __lt__(self, other):
    return Expr(taichi_lang.expr_cmp_lt(self.ptr, other.ptr))

  def __ge__(self, other):
    return Expr(taichi_lang.expr_cmp_ge(self.ptr, other.ptr))

  def __gt__(self, other):
    return Expr(taichi_lang.expr_cmp_gt(self.ptr, other.ptr))

  def __eq__(self, other):
    return Expr(taichi_lang.expr_cmp_eq(self.ptr, other.ptr))

  def __ne__(self, other):
    return Expr(taichi_lang.expr_cmp_ne(self.ptr, other.ptr))

  def __getitem__(self, item):
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

def comp(foo):
  src = inspect.getsource(foo)
  tree = ast.parse(src)
  # print(astor.to_source(tree.body[0]))

  func_body = tree.body[0]

  visitor = ASTTransformer()
  visitor.visit(tree)
  ast.fix_missing_locations(tree)

  # astpretty.pprint(func_body)
  #print(codegen.to_source(tree))
  print(astor.to_source(tree.body[0]))

  exec(compile(tree, filename='tmp', mode='exec'))
  return locals()[foo.__name__]
  #print(locals())
  #print(globals())
  #return locals()[foo.__name__]

def test():
  a = Expr(1)
  b = Expr(2)
  c = a + b
  print(c.serialize())

test = comp(test)


if __name__ == '__main__':
  prog = taichi_lang.Program()
  a = Expr(taichi_lang.make_id_expr(""))
  a.ptr = taichi_lang.global_new(a.ptr, taichi_lang.DataType.float32)
  def l():
    for i in range(10):
      taichi_lang.get_root().place(a.ptr)
  taichi_lang.layout(l)
  test()
