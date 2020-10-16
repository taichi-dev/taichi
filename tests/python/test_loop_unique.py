import taichi as ti


@ti.test(require=ti.extension.sparse)
def test_loop_unique_simple_1d():
    x, y = ti.field(ti.i32), ti.field(ti.i32)

    N = 16
    ti.root.pointer(ti.i, N).place(x)
    ti.root.pointer(ti.i, N).place(y)

    @ti.kernel
    def inc_y():
        for i in x:
            a = ti.loop_unique(x[i])
            y[a] += 1

    x[1] = 2
    x[2] = 3
    x[7] = 5
    y[3] = 2
    y[4] = 3
    inc_y()
    expected_result = {2: 1, 3: 3, 4: 3, 5: 1}
    for i in range(N):
        assert y[i] == expected_result.get(i, 0)


@ti.test(require=ti.extension.sparse)
def test_loop_unique_nested_1d():
    x, y = ti.field(ti.i32), ti.field(ti.i32)

    N = 16
    ti.root.pointer(ti.i, N).place(x)
    ti.root.pointer(ti.i, N).place(y)

    @ti.kernel
    def inc_y():
        for i in x:
            for j in range(i):
                a = ti.loop_unique(x[i])
                y[a] += 1

    x[1] = 2
    x[2] = 3
    x[7] = 5
    y[3] = 2
    y[4] = 3
    inc_y()
    expected_result = {2: 1, 3: 4, 4: 3, 5: 7}
    for i in range(N):
        assert y[i] == expected_result.get(i, 0)


@ti.test(require=ti.extension.sparse)
def test_loop_unique_2d():
    x, y, z = ti.field(ti.i32), ti.field(ti.i32), ti.field(ti.i32)

    N = 8
    ti.root.pointer(ti.ij, N).place(x)
    ti.root.pointer(ti.ij, N).place(y)
    ti.root.pointer(ti.ij, N).place(z)

    @ti.kernel
    def inc_y_z():
        for i, j in x:
            a = ti.loop_unique(x[i, j])
            y[a, j] += 1
            z[i, i] += 1  # cannot demote this

    x[1, 1] = 2
    x[1, 2] = 4
    x[1, 3] = 5
    x[1, 4] = 7
    x[1, 5] = 0
    x[1, 6] = 1
    x[2, 5] = 3
    x[2, 7] = 6
    y[3, 5] = 3
    y[6, 6] = 8
    z[2, 2] = 5
    inc_y_z()
    expected_result_y = {
        (0, 5): 1,
        (1, 6): 1,
        (2, 1): 1,
        (3, 5): 4,
        (4, 2): 1,
        (5, 3): 1,
        (6, 6): 8,
        (6, 7): 1,
        (7, 4): 1
    }
    expected_result_z = {(1, 1): 6, (2, 2): 7}
    for i in range(N):
        for j in range(N):
            assert y[i, j] == expected_result_y.get((i, j), 0)
            assert z[i, j] == expected_result_z.get((i, j), 0)
