import taichi as ti

ti.init(arch=ti.opengl, log_level=ti.DEBUG)

x = ti.var(ti.f32, ())


@ti.kernel
def func(a: ti.f32, b: ti.f32):
    t = 0.0
    ti.asm('$0 = %0 * %1', inputs=[a, b], outputs=[t])
    x[None] = t


func(3, 4)
print(x[None])
