import inspect
import astpretty
import ast
import taichi as tc

def foo(x):
  a[i] += 1

src = inspect.getsource(foo)
tree = ast.parse(src)
astpretty.pprint(tree)


