import taichi as ti


@ti.all_archs
def test_advanced_store_forwarding_nested_loops():
    val = ti.var(ti.i32)
    ti.root.place(val)

    @ti.kernel
    def func():
        # If we want to do store-forwarding to local loads inside loops,
        # we should pass the last local store into the loop, rather than use
        # an empty AllocaOptimize loop.
        # See https://github.com/taichi-dev/taichi/pull/849.
        a = val[None]
        for i in range(1):
            for j in range(1):
                val[None] = a

    val[None] = 10
    func()
    assert val[None] == 10
