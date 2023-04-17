import taichi as ti

ti.init()

x = ti.field(dtype=ti.i32)
block1 = ti.root.pointer(ti.ij, (4, 4))
block2 = block1.pointer(ti.ij, (2, 2))
pixel = block2.dense(ti.ij, (2, 2))
pixel.place(x)


@ti.kernel
def sparse_api_demo():
    ti.activate(block1, [0, 1])
    ti.activate(block2, [1, 2])

    for i, j in x:
        print(f"field x[{i}, {j}] = {x[i, j]}")
    # outputs:
    # field x[2, 4] = 0
    # field x[2, 5] = 0
    # field x[3, 4] = 0
    # field x[3, 5] = 0

    for i, j in block2:
        print(f"Active block2: [{i}, {j}]")
    # output: Active block2: [1, 2]

    for i, j in block1:
        print(f"Active block1: [{i}, {j}]")
    # output: Active block1: [0, 1]

    for j in range(4):
        print(f"Activity of block2[2, {j}] = {ti.is_active(block2, [1, j])}")

    ti.deactivate(block2, [1, 2])

    for i, j in block2:
        print(f"Active block2: [{i}, {j}]")
    # output: nothing

    for i, j in block1:
        print(f"Active block1: [{i}, {j}]")
    # output: Active block1: [0, 1]

    print(ti.rescale_index(x, block1, ti.Vector([9, 17])))
    # output = [2, 4]

    # Note: ti.Vector is optional in ti.rescale_index.
    print(ti.rescale_index(x, block1, [9, 17]))
    # output = [2, 4]

    ti.activate(block2, [1, 2])


sparse_api_demo()


@ti.kernel
def check_activity(snode: ti.template(), i: ti.i32, j: ti.i32):
    print(ti.is_active(snode, [i, j]))


check_activity(block2, 1, 2)  # output = 1
block2.deactivate_all()
check_activity(block2, 1, 2)  # output = 0
check_activity(block1, 0, 1)  # output = 1
ti.deactivate_all_snodes()
check_activity(block1, 0, 1)  # output = 0
