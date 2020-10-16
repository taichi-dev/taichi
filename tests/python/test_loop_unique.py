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
