import taichi as ti

ti.init(arch=ti.cpu, debug=True, print_ir=True)

x = ti.field(ti.i32)

L = ti.root.pointer(ti.ij, 32)
L.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

for i in range(1024):
    x[i * 8, i * 8] = 1
    L.deactivate_all()
    ti.memory_profiler_print()
