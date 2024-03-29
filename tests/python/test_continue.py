import taichi as ti
from tests import test_utils

n = 1000


@test_utils.test()
def test_for_continue():
    x = ti.field(ti.i32, shape=n)

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


@test_utils.test()
def test_while_continue():
    x = ti.field(ti.i32, shape=n)

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


@test_utils.test()
def test_kernel_continue():
    x = ti.field(ti.i32, shape=n)

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


@test_utils.test()
def test_unconditional_continue():
    x = ti.field(ti.i32, shape=n)

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


@test_utils.test()
def test_kernel_continue_in_nested_if():
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def run(a: ti.i32):
        for i in range(1):
            if a:
                if a:
                    continue
            if a:
                if a:
                    continue
            x[i] = i

    x[0] = 1
    run(1)
    assert x[0] == 1
    run(0)
    assert x[0] == 0


@test_utils.test()
def test_kernel_continue_in_nested_if_2():
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def run(a: ti.i32):
        for i in range(1):
            if a:
                if a:
                    continue
            if a:
                continue
            x[i] = i

    x[0] = 1
    run(1)
    assert x[0] == 1
    run(0)
    assert x[0] == 0


@test_utils.test()
def test_kernel_continue_in_nested_if_3():
    x = ti.field(ti.i32, shape=n)

    @ti.kernel
    def run(a: ti.i32):
        for i in range(1):
            if a:
                continue
            if a:
                if a:
                    continue
            x[i] = i

    x[0] = 1
    run(1)
    assert x[0] == 1
    run(0)
    assert x[0] == 0


@test_utils.test()
def test_kernel_continue_in_simple_if():
    img = ti.field(ti.i32, (2, 2))

    @ti.kernel
    def K():
        for i, j in img:
            img[i, j] = 0
            if i > 0 or j > 0:
                continue
            img[i, j] = 1

    img.fill(2)
    K()

    for i in range(2):
        for j in range(2):
            if i > 0 or j > 0:
                assert img[i, j] == 0
            else:
                assert img[i, j] == 1
