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
print(taichi_lang.BinaryType.mul)

x = []

# Translate AST into taichi lang AST
# Grammar: https://docs.python.org/3/library/ast.html#abstract-grammar

a = ast.BoolOp()

class FuncVisitor(ast.NodeVisitor):
  def __init__(self):
    self.indent = 0
    self.inside_statement = False

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

  def visit_Name(self, node):
    pass

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
  for i in x:
    x[i] += i
  for i in x:
    x[i] += i * 2

