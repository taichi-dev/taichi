import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.f32)
z, w = ti.var(ti.f32), ti.var(ti.f32)

ti.cfg.use_llvm = True
ti.cfg.print_ir = True
ti.runtime.print_preprocessed = True

@ti.layout
def xy():
  fork = ti.root.dense(ti.k, 128)
  fork.dense(ti.ij, 16).place(x, y)
  fork.dense(ti.ijk, 4).dense(ti.i, 8).place(z, w)

@ti.kernel
def test():
  a = 0
  for i in range(10):
    if i % 2 == 0:
      a += i
  ti.print(a)

test()

