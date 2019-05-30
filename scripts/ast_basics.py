import inspect
import astpretty
import ast

x = []

class FuncVisitor(ast.NodeVisitor):
  def __init__(self):
    self.indent = 0

  def generic_visit(self, node):
    for i in range(self.indent):
      print('  ', end='')
    print(type(node).__name__)
    self.indent += 1
    ast.NodeVisitor.generic_visit(self, node)
    self.indent -= 1

def taichi(foo):
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

@taichi
def foo():
  for i in x:
    x[i] += i
  for i in x:
    x[i] += i * 2

