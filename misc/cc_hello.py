import taichi as ti

ti.init(arch=ti.cc)
ti.core.toggle_advanced_optimization(False)

x = ti.var(ti.i32)
y = ti.var(ti.f32)
z = ti.var(ti.f64)

blk0 = ti.root
blk1 = blk0.dense(ti.i, 8)
blk2 = blk1.dense(ti.i, 4)
blk1.place(x)
blk1.place(y)
blk2.place(z)


@ti.kernel
def func():
    z[1] += 2.4
    z[3] = 4.2 + z[1]
    print('Nice to meet you!', min(z[3], 9))


func()
