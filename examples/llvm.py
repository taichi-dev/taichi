import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.f32)

ti.cfg.use_llvm = True
ti.cfg.print_ir = True

@ti.layout
def xy():
  ti.root.dense(ti.ij, 16).place(x, y)

@ti.kernel
def test():
  # i = 1
  # a = i + i
  ti.print(1)


test()

