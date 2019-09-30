import taichi_lang as ti

# ti.cfg.print_ir = True
ti.cfg.arch = ti.cuda
x = ti.var(ti.f32)

@ti.layout
def X():
  ti.root.place(x)

@ti.kernel
def kernel(x: ti.i32):
  ti.print(x)

@ti.kernel
def kernel2(x: ti.i32, y: ti.f32):
  ti.print(x + y)

kernel(123)
kernel(456)

kernel2(456, 0.7)
