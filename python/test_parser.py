import inspect
import astpretty
import ast
import taichi as tc

def taichi(**kwargs):
  return lambda x: x


@taichi(dim=2, opt=False)
def foo(x):
  #ttt.print(1)
  for i in range(1, 10):
    pass

  a.b = 1

  if i == 0:
    a()
  else:
    b()

# p = ast.parse(expr)
# p.body[0].body = [ast.parse("return 42").body[0]] # Replace function body with "return 42"
# astpretty.pprint(p)
src = inspect.getsource(foo)
tree = ast.parse(src)
# print(tree)
astpretty.pprint(tree)


@taichi
def main():
  for p in range(10):
    for i, j in x:
      for k in range(dim):
        x(k)[i, j] += 123

  for k in range(dim):
    for i, j in ti.range(x):
        x(k)[i, j] += 123


main()
