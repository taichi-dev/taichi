import taichi as ti


@ti.all_archs
def test_offload_order():
    n = 128
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    y = ti.field(ti.f32, shape=n, needs_grad=True)
    z = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def forward():
        for i in x:
            y[i] = x[i]

        # for i in x:
        #     z[None] += y[i]

    with ti.Tape(z):
        forward()

    # for i in range(n):
    #     assert x.grad[i] == 1
