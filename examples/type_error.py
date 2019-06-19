import taichi_lang as ti

x, y = ti.var(ti.f32), ti.var(ti.i32)

@ti.layout
def xy():
  ti.root.place(x, y)

@ti.kernel
def trigger():
  z = x + y

trigger()

