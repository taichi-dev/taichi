import taichi as ti


@ti.all_archs_with(print_ir=True)
def test_for_break():
    return
    x = ti.var(ti.f32)

    N = 16

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(N + 5):
                if j >= i:
                    break
                x[i] = j

    func()
    for i in range(N):
        assert x[i] == i
