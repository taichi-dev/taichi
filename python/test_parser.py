import inspect
import astpretty
import ast
import taichi as tc

def taichi(**kwargs):
  return lambda x: x


@taichi(dim=2, opt=False)
def foo(x):
  #ttt.print(1)
  # for i in range(1, 10):
  #  pass

  a[i] = 1
  a[i] = b[i, j]
  c[i, j, k] = 123

  '''
  a = ti.float32(123)

  x = float32(1)
  x = float64(1)
  x = int(1)
  y = x
  x[i] = (1 + 2 - 3)

  if dim == 2:
    print(2)
  else:
    print(1)

  if x[i, j] == 2:
    pass
  else:
    pass

  unroll()
  for i, j in range(x):
    print("hello world")
    print(x[i, j])
    SLP(4)

  for t in range(100):
    for i in ti.range(123):
      pass
    pass
  '''


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
