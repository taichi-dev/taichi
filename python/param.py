import taichi_lang as ti

ti.cfg.print_ir = True
x = ti.var(ti.f32)

@ti.layout
def X():
  ti.root.place(x)

@ti.kernel
def kernel(x: ti.i32):
  ti.print(x)

kernel(123)
