import taichi as ti

use_bitmask = False

ti.init()

x = ti.field(dtype=ti.i32)
block = ti.root.pointer(ti.ij, (4, 4))
if use_bitmask:
    pixel = block.bitmasked(ti.ij, (2, 2))
else:
    pixel = block.dense(ti.ij, (2, 2))
pixel.place(x)


@ti.kernel
def sparse_struct_for():
    x[2, 3] = 2
    x[5, 6] = 3

    for i, j in x:
        print(f"field x[i, j]={x[i, j]}")

    for i, j in block:
        print(f"Active block: [{i}, {j}]")


print(f"use_bitmask={use_bitmask}")
sparse_struct_for()
