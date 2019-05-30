import inspect
import astpretty
import ast
import taichi as tc
from core import taichi_lang

taichi_lang.lang()

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
    for _ in range(self.indent):
      print('  ', end='')
    print(type(node).__name__)
    self.indent += 1
    ast.NodeVisitor.generic_visit(self, node)
    self.indent -= 1

  def visit_For(self, node):
    print(node.target)
    iter = node.iter
    is_taichi_loop = False

    if isinstance(iter, ast.Call):
      if iter.func.id == 'trange':
        is_taichi_loop = True

    self.generic_visit(node)

  def visit_AugAssign(self, node):
    self.generic_visit(node)

  def visit_Tuple(self, node):
    self.generic_visit(node)
    g = taichi_lang.ExprGroup()
    for s in node.elts:
      g.push_back(self.exprs[s])
    self.exprs[node] = g
    print(g.serialize())

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

  def visit_UnaryOp(self, node):
    self.generic_visit(node)
    if isinstance(node.op, ast.UAdd):
      self.exprs[node] = self.exprs[node.operand]
      return
    if isinstance(node.op, ast.USub):
      op = taichi_lang.UnaryOpType.neg
    else:
      assert False
    expr = taichi_lang.make_unary_op_expr(op, self.exprs[node.operand])
    self.exprs[node] = expr
    print(expr.serialize())

  def visit_Name(self, node):
    self.exprs[node] = taichi_lang.make_id_expr(node.id)

  def visit_Num(self, node):
    self.generic_visit(node)
    self.exprs[node] = taichi_lang.make_constant_expr(node.n)

  def visit_Subscript(self, node):
    self.generic_visit(node)
    self.exprs[node] = taichi_lang.make_global_ptr_expr(self.exprs[node.value], self.exprs[node.slice.value])
    print(self.exprs[node].serialize())

  def visit_Index(self, node):
    self.generic_visit(node)
    self.exprs[node] = self.exprs[node.value]

def ti(foo):
  src = inspect.getsource(foo)
  tree = ast.parse(src)
  # astpretty.pprint(tree)

  func_body = tree.body[0]
  statements = func_body.body

  for s in statements:
    astpretty.pprint(s)

  visitor = FuncVisitor()
  visitor.visit(tree)

  return foo

