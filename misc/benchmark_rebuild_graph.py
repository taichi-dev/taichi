import taichi as ti
from taichi.lang import impl

ti.init(arch=ti.cuda, async_mode=True)

a = ti.field(dtype=ti.i32, shape=())
b = ti.field(dtype=ti.i32, shape=())
c = ti.field(dtype=ti.i32, shape=())
d = ti.field(dtype=ti.i32, shape=())
e = ti.field(dtype=ti.i32, shape=())
f = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def foo():
    a[None] += 1
    b[None] += 1
    c[None] += 1
    d[None] += 1
    e[None] += 1
    f[None] += 1


for i in range(1000):
    foo()

impl.get_runtime().prog.benchmark_rebuild_graph()
