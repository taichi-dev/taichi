import inspect
import astpretty
import ast
import taichi as tc
import os
try:
  os.symlink(tc.get_build_directory() + '/libtaichi_lang.so', tc.get_build_directory() + '/taichi_lang.so')
except:
  pass
import taichi_lang

taichi_lang.lang()
print(taichi_lang.BinaryOpType.mul)
one = taichi_lang.make_constant_expr(1)
two = taichi_lang.make_constant_expr(2)
expr = taichi_lang.make_binary_op_expr(taichi_lang.BinaryOpType.add, one, two)
print(expr.serialize())

x = []

# Translate AST into taichi lang AST
# Grammar: https://docs.python.org/3/library/ast.html#abstract-grammar

a = ast.BoolOp()

class FuncVisitor(ast.NodeVisitor):
  def __init__(self):
    self.indent = 0
    self.inside_statement = False
    self.exprs = {}

  def generic_visit(self, node):
    for i in range(self.indent):
      print('  ', end='')
    print(type(node).__name__)
    self.indent += 1
    ast.NodeVisitor.generic_visit(self, node)
    self.indent -= 1

  def visit_For(self, node):
    self.generic_visit(node)


  def visit_AugAssign(self, node):
    self.generic_visit(node)

  # differentiate visit statement and visit expr

  def visit_BinOp(self, node):
    self.generic_visit(node)
    if isinstance(node.op, ast.Add):
      op = taichi_lang.BinaryOpType.add
    elif isinstance(node.op, ast.Sub):
      op = taichi_lang.BinaryOpType.sub
    elif isinstance(node.op, ast.Mult):
      op = taichi_lang.BinaryOpType.mul
    elif isinstance(node.op, ast.Div):
      op = taichi_lang.BinaryOpType.div
    elif isinstance(node.op, ast.Mod):
      op = taichi_lang.BinaryOpType.mod
    elif isinstance(node.op, ast.BitOr):
      op = taichi_lang.BinaryOpType.bit_or
    elif isinstance(node.op, ast.BitXor):
      op = taichi_lang.BinaryOpType.bit_xor
    else:
      assert False
    expr = taichi_lang.make_binary_op_expr(op, self.exprs[node.left], self.exprs[node.right])
    self.exprs[node] = expr
    print(expr.serialize())

  def visit_Name(self, node):
    pass

  def visit_Num(self, node):
    self.generic_visit(node)
    self.exprs[node] = taichi_lang.make_constant_expr(node.n)

def ti(foo):
  src = inspect.getsource(foo)
  tree = ast.parse(src)
  astpretty.pprint(tree)

  func_body = tree.body[0]
  statements = func_body.body

  for s in statements:
    astpretty.pprint(s)

  visitor = FuncVisitor()
  visitor.visit(tree)

  return foo

@ti
def foo():
  #for i in x:
  #  x[i] += i
  for i in x:
    x[i] += 1 + 2 + 3 * 4 / 5 - 6

