import taichi_lang as ti
from pytest import approx

# ti.cfg.print_ir = True

def test_complex_kernels():
  for arch in [ti.x86_64]:
    ti.reset()
    ti.cfg.arch = arch

    a = ti.var(ti.f32)
    b = ti.var(ti.f32)

    n = 128

    @ti.layout
    def place():
      ti.root.dense(ti.i, n).place(a, b)

    @ti.kernel
    def add():
      for i in range(n):
        a[i] += 1
      for i in range(n):
        b[i] += 2
      for i in a:
        b[i] += 3

    for i in range(n):
      a[i] = i + 1
      b[i] = i + 2
    add()

    for i in range(n):
      assert a[i] == i + 2
      assert b[i] == i + 7

# test_complex_kernels()
