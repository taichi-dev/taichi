import taichi as ti


@ti.all_archs
def test_kernel_sync():
    n = 128
    x = ti.field(ti.i32, shape=(3, ))
    y = ti.field(ti.i32, shape=(n, ))
    # These [] calls are all on CPU, so no synchronization needed
    x[0] = 42
    assert x[0] == 42
    x[1] = 233
    x[2] = -1

    @ti.kernel
    def func():
        for i in y:
            y[i] = x[i % 3]

    # Kernel *may* run on GPU
    # Note that the previous kernel is a write, which didn't do a sync. But that
    # should be fine -- we only need to sync the memory after GPU -> CPU.
    func()
    # These [] calls are on CPU. They should be smart enough to sync only once.
    for i in range(n):
        assert y[i] == x[i % 3]
