# This file is not part of standard tests since it uses too much GPU memory

import taichi as ti

ti.init(arch=ti.cuda, debug=True)

res = 512

mask = ti.field(ti.i32)
val = ti.field(ti.f32)

ti.root.dense(ti.ijk, 512).place(mask)
block = ti.root.pointer(ti.ijk, 128).dense(ti.ijk, 4)
block.dense(ti.l, 128).place(val)


@ti.kernel
def load_inputs():
    for i, j, k in mask:
        for l in range(128):
            val[i, j, k, l] = 1


load_inputs()
