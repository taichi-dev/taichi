import taichi as ti

@ti.program_test
def test_complex_kernels():
  for arch in [ti.x86_64, ti.cuda]:
    ti.reset()
    ti.cfg.arch = arch
    if ti.cfg.use_llvm:
      # LLVM currently does not support this...
      continue

    a = ti.var(ti.f32)
    b = ti.var(ti.f32)

    n = 128

    @ti.layout
    def place():
      ti.root.dense(ti.i, n).place(a, b)

    # Note: the CUDA backend can indeed translate this into multiple kernel launches
    @ti.kernel
    def add():
      for i in range(n):
        a[i] += 1
      for i in range(n):
        b[i] += 2
      for i in a:
        b[i] += 3
      for i in b:
        a[i] += 1
      for i in a:
        a[i] += 9

    for i in range(n):
      a[i] = i + 1
      b[i] = i + 2
    add()

    for i in range(n):
      assert a[i] == i + 12
      assert b[i] == i + 7

