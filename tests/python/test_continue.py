import taichi as ti

n = 1000


@ti.all_archs
def test_for_continue():
    x = ti.var(ti.i32, shape=n)

    @ti.kernel
    def run():
        # Launch just one thread
        for _ in range(1):
            for j in range(n):
                if j % 2 == 0:
                    continue
                x[j] = j

    run()
    xs = x.to_numpy()
    for i in range(n):
        expect = 0 if i % 2 == 0 else i
        assert xs[i] == expect


@ti.all_archs
def test_while_continue():
    x = ti.var(ti.i32, shape=n)

    @ti.kernel
    def run():
        # Launch just one thread
        for _ in range(1):
            j = 0
            while j < n:
                oj = j
                j += 1
                if oj % 2 == 0:
                    continue
                x[oj] = oj

    run()
    xs = x.to_numpy()
    for i in range(n):
        expect = 0 if i % 2 == 0 else i
        assert xs[i] == expect


@ti.all_archs
def test_kernel_continue():
    x = ti.var(ti.i32, shape=n)

    @ti.kernel
    def run():
        for i in range(n):
            if i % 2 == 0:
                # At kernel level, this is the same as return
                continue
            x[i] = i

    run()
    xs = x.to_numpy()
    for i in range(n):
        expect = 0 if i % 2 == 0 else i
        assert xs[i] == expect


@ti.all_archs
def test_unconditional_continue():
    x = ti.var(ti.i32, shape=n)

    @ti.kernel
    def run():
        # Launch just one thread
        for _ in range(1):
            for j in range(n):
                continue
                # pylint: disable=unreachable
                x[j] = j

    run()
    xs = x.to_numpy()
    for i in range(n):
        assert xs[i] == 0
