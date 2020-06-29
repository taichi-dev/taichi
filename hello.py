import taichi as ti

ti.init(arch=ti.cc)

x = ti.var(ti.i32)
y = ti.var(ti.f32)
z = ti.var(ti.f64)

blk0 = ti.root
blk1 = blk0.dense(ti.i, 8)
blk2 = blk1.dense(ti.j, 4)
blk1.place(x)
blk1.place(y)
blk2.place(z)

@ti.kernel
def func():
    print('Hello world!')
    print('Nice to meet you!', 233)

func()
