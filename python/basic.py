import taichi_lang as ti

x = ti.global_var(ti.i32)

def sum(a, b):
  return a + b

@ti.kernel
def test():
  a = 1
  b = 2
  c = a + b * b + sum(a, b)
  c = c + 1
  for i in range(0, 10):
    c = c + i
    x[i + 3, i + 3] = i
    ti.print(x[i + 2, i + 2])

  ti.print(c)
  ti.print(1)
  ti.print(1 + 11)

@ti.layout
def place_variables():
  ti.root.dense(ti.indices(0, 1), (256, 256)).place(x.ptr)

test()
