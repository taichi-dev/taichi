import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.f32)

ti.cfg.use_llvm = True
ti.cfg.print_ir = True
ti.runtime.print_preprocessed = True

@ti.layout
def xy():
  ti.root.dense(ti.ij, 16).place(x, y)

@ti.kernel
def test():
  # i = 1
  a = 42
  a += 1
  for i in range(16):
    a += 1
  ti.print(a)


test()

