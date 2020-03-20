import taichi as ti


@ti.all_archs_with(print_ir=True)
def test_for_break():
    x = ti.var(ti.i32)
    N, M = 4, 4
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(M):
                if j > i: break
                x[i, j] = 100 * i + j

    func()
    for i in range(N):
        for j in range(M):
            if j > i:
                assert x[i, j] == 0
            else:
                assert x[i, j] == 100 * i + j
