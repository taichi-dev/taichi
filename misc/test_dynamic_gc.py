import taichi as ti

ti.init(arch=ti.cpu, debug=True)

x = ti.field(ti.i32)

L = ti.root.dynamic(ti.i, 1024 * 1024, chunk_size=1024)
L.place(x)

while True:
    x[1024] = 1
    L.deactivate_all()
    ti.memory_profiler_print()
