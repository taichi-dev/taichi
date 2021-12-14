import taichi as ti


@ti.test(exclude=[ti.cc])
def test_sort():
    def test_sort_for_dtype(dtype, N):
        x = ti.field(dtype, N)

        @ti.kernel
        def fill():
            for i in x:
                x[i] = ti.random() * N

        fill()
        ti.parallel_sort(x)

        x_host = x.to_numpy()

        for i in range(N - 1):
            assert x_host[i] <= x_host[i + 1]

    test_sort_for_dtype(ti.i32, 1)
    test_sort_for_dtype(ti.i32, 256)
    test_sort_for_dtype(ti.i32, 100001)
    test_sort_for_dtype(ti.f32, 1)
    test_sort_for_dtype(ti.f32, 256)
    test_sort_for_dtype(ti.f32, 100001)
