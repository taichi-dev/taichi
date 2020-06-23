import taichi as ti

ti.init(arch=ti.cuda, debug=True)

res = 512

mask = ti.var(ti.i32)
F_in = ti.var(ti.f32)

ti.root.dense(ti.ijk, 512).place(mask)
block = ti.root.pointer(ti.ijk, 128).dense(ti.ijk, 4)
block.dense(ti.l, 128).place(F_in)


@ti.kernel
def load_inputs():
    for i, j, k in mask:
        for l in range(128):
            F_in[i, j, k, l] = 1


load_inputs()