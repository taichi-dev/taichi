import inspect
import astpretty
import ast
import taichi as tc

def foo(x):
  global y
  def pr(x: ti.i32):
    print(x)

src = inspect.getsource(foo)
tree = ast.parse(src)
astpretty.pprint(tree)

