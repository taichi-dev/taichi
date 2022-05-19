import taichi as ti
from tests import test_utils

n = 128


def run_atomic_add_global_case(vartype, step, valproc=lambda x: x):
    x = ti.field(vartype)
    y = ti.field(vartype)
    c = ti.field(vartype)

    ti.root.dense(ti.i, n).place(x, y)
    ti.root.place(c)

    # Make Taichi correctly infer the type
    # TODO: Taichi seems to treat numpy.int32 as a float type, fix that.
    init_ck = 0 if vartype == ti.i32 else 0.0

    @ti.kernel
    def func():
        ck = init_ck
        for i in range(n):
            x[i] = ti.atomic_add(c[None], step)
            y[i] = ti.atomic_add(ck, step)

    func()

    assert valproc(c[None]) == n * step
    x_actual = sorted(x.to_numpy())
    y_actual = sorted(y.to_numpy())
    expect = [i * step for i in range(n)]
    for (xa, ya, e) in zip(x_actual, y_actual, expect):
        print(xa, ya, e)
        assert valproc(xa) == e
        assert valproc(ya) == e


@test_utils.test()
def test_atomic_add_global_i32():
    run_atomic_add_global_case(ti.i32, 42)


@test_utils.test()
def test_atomic_add_global_f32():
    run_atomic_add_global_case(
        ti.f32, 4.2, valproc=lambda x: test_utils.approx(x, rel=1e-5))


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_atomic_min_max_uint():
    x = ti.field(ti.u64, shape=100)

    @ti.kernel
    def test0():
        for I in x:
            x[I] = 0
        x[1] = ti.cast(1, ti.u64) << 63
        for I in x:
            ti.atomic_max(x[0], x[I])

    test0()
    assert x[0] == 9223372036854775808

    @ti.kernel
    def test1():
        for I in x:
            x[I] = ti.cast(1, ti.u64) << 63
        x[1] = 100
        for I in x:
            ti.atomic_min(x[0], x[I])

    test1()
    assert x[0] == 100


@test_utils.test()
def test_atomic_add_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_add(c[None], step)

    func()

    assert c[None] == n * step


@test_utils.test()
def test_atomic_add_demoted():
    # Ensure demoted atomics do not crash the program.
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def func():
        for i in range(n):
            s = i
            # Both adds should get demoted.
            x[i] = ti.atomic_add(s, step)
            y[i] = ti.atomic_add(s, step)

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + step


@test_utils.test()
def test_atomic_add_with_local_store_simplify1():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    # local store [$a <- ...]
    #
    # Specifically, the second store should not suppress the first one, because
    # atomic_add can return value.
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def func():
        for i in range(n):
            # do a local store
            j = i
            x[i] = ti.atomic_add(j, step)
            # do another local store, make sure the previous one is not optimized out
            j = x[i]
            y[i] = j

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i


@test_utils.test()
def test_atomic_add_with_local_store_simplify2():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    #
    # Specifically, the local store should not be removed, because
    # atomic_add can return its value.
    x = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x)

    @ti.kernel
    def func():
        for i in range(n):
            j = i
            x[i] = ti.atomic_add(j, step)

    func()

    for i in range(n):
        assert x[i] == i


@test_utils.test()
def test_atomic_add_with_if_simplify():
    # Make sure IfStmt simplification doesn't move stmts depending on the result
    # of atomic_add()
    x = ti.field(ti.i32)
    step = 42

    ti.root.dense(ti.i, n).place(x)

    boundary = n / 2

    @ti.kernel
    def func():
        for i in range(n):
            if i > boundary:
                # A sequence of commands designed such that atomic_add() is the only
                # thing to decide whether the if branch can be simplified.
                s = i
                j = ti.atomic_add(s, s)
                k = j + s
                x[i] = k
            else:
                # If we look at the IR, this branch should be simplified, since nobody
                # is using atomic_add's result.
                ti.atomic_add(x[i], i)
                x[i] += step

    func()

    for i in range(n):
        expect = i * 3 if i > boundary else (i + step)
        assert x[i] == expect


@test_utils.test()
def test_local_atomic_with_if():
    ret = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def test():
        if True:
            x = 0
            x += 1
            ret[None] = x

    test()
    assert ret[None] == 1


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_atomic_sub_with_type_promotion():
    # Test Case 1
    @ti.kernel
    def test_u16_sub_u8() -> ti.uint16:
        x: ti.uint16 = 1000
        y: ti.uint8 = 255

        ti.atomic_sub(x, y)
        return x

    res = test_u16_sub_u8()
    assert res == 745

    # Test Case 2
    @ti.kernel
    def test_u8_sub_u16() -> ti.uint8:
        x: ti.uint8 = 255
        y: ti.uint16 = 100

        ti.atomic_sub(x, y)
        return x

    res = test_u8_sub_u16()
    assert res == 155

    # Test Case 3
    A = ti.field(ti.uint8, shape=())
    B = ti.field(ti.uint16, shape=())

    @ti.kernel
    def test_with_field():
        v: ti.uint16 = 1000
        v -= A[None]
        B[None] = v

    A[None] = 255
    test_with_field()
    assert B[None] == 745


@test_utils.test()
def test_atomic_sub_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_sub(c[None], step)

    func()

    assert c[None] == -n * step


@test_utils.test()
def test_atomic_max_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_max(c[None], i * step)

    func()

    assert c[None] == (n - 1) * step


@test_utils.test()
def test_atomic_min_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 1000
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_min(c[None], i * step)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_and_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    max_int = 2147483647

    @ti.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_and(c[None], max_int - 2**i)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_or_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 0
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_or(c[None], 2**i)

    func()

    assert c[None] == 1023


@test_utils.test()
def test_atomic_xor_expr_evaled():
    c = ti.field(ti.i32)
    step = 42

    ti.root.place(c)

    @ti.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            ti.atomic_xor(c[None], 2**i)

    func()

    assert c[None] == 0
