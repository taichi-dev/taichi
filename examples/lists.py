import taichi_lang as ti

x = ti.global_var(ti.i32)
n = 16

@ti.layout
def lists():
  ti.root.dense(ti.i, n).dynamic(ti.j, n).place(x)


@ti.kernel
def make_lists():
  for i in range(n):
    for j in range(i):
      ti.append(x.parent(), i, j * j)

make_lists()

for i in range(n):
  for j in range(n):
    assert x[i, j] == (j * j if j < i else 0)