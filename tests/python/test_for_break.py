import taichi as ti


@ti.all_archs_with(print_ir=True)
def test_for_break():
    x = ti.Vector(2, dt=ti.f32)
    N, M = 16, 16
    ti.root.dense(ti.ij, (N, M)).place(x)

    @ti.kernel
    def func():
        for i in range(N):
            for j in range(M):
                if j > i: break
                x[i, j] = [i, j]

    func()
    print(x.to_numpy())
