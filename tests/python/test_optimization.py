import taichi as ti


@ti.all_archs
def test_advanced_store_forwarding_nested_loops():
    val = ti.field(ti.i32)
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


@ti.all_archs
def test_advanced_unused_store_elimination_if():
    val = ti.field(ti.i32)
    ti.root.place(val)

    @ti.kernel
    def func():
        a = 1
        if val[None]:
            a = 2
            if val[None]:
                a = 3
            else:
                a = 4
            val[None] = a
        else:
            val[None] = a

    val[None] = 0
    func()
    assert val[None] == 1
    func()
    assert val[None] == 3


@ti.all_archs
def test_local_store_in_nested_for_and_if():
    # See https://github.com/taichi-dev/taichi/pull/862.
    val = ti.field(ti.i32, shape=(3, 3, 3))

    @ti.kernel
    def func():
        ti.serialize()
        for i, j, k in val:
            if i < 2 and j < 2 and k < 2:
                a = 0
                for di, dj, dk in ti.ndrange((0, 2), (0, 2), (0, 2)):
                    if val[i + di, j + dj, k + dk] == 1:
                        a = val[i + di, j + dj, k + dk]

                for di, dj, dk in ti.ndrange((0, 2), (0, 2), (0, 2)):
                    val[i + di, j + dj, k + dk] = a

    val[1, 1, 1] = 1
    func()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert (val[i, j, k] == 1)


@ti.all_archs
def test_advanced_store_forwarding_continue_in_if():
    val = ti.field(ti.i32)
    ti.root.place(val)

    @ti.kernel
    def func(n: ti.i32):
        # Launch just one thread
        for _ in range(1):
            a = 10
            b = 0
            for i in range(n):
                b += a
                a = i
                if i == 5:
                    continue
                a = 100
            a = 1000
            val[None] = a + b

    func(1)
    assert val[None] == 1010
    func(5)
    assert val[None] == 1410
    func(6)
    assert val[None] == 1510
    func(7)
    assert val[None] == 1515


@ti.all_archs
def test_advanced_store_elimination_in_loop():
    val = ti.field(ti.i32)
    ti.root.place(val)

    @ti.kernel
    def func():
        # Launch just one thread
        for _ in range(1):
            a = 1
            for i in range(5):
                b = 1
                val[None] = a + b
                b = 0
                a = 2
                a = 3
            a = 4
            val[None] += a

    func()
    assert val[None] == 8


@ti.all_archs
def test_parallel_assignment():
    mat = ti.field(ti.i32, shape=(3, 4))

    @ti.kernel
    def func():
        c = 0
        for i in ti.static(range(4)):
            mat[0, c], mat[1, c], mat[2, c] = 1, 2, 3
            c += 1

    func()
    for i in range(3):
        for j in range(4):
            assert mat[i, j] == i + 1
