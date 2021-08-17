import taichi as ti
from taichi import approx

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


@ti.all_archs
def test_atomic_add_global_i32():
    run_atomic_add_global_case(ti.i32, 42)


@ti.all_archs
def test_atomic_add_global_f32():
    run_atomic_add_global_case(ti.f32,
                               4.2,
                               valproc=lambda x: approx(x, rel=1e-5))


@ti.all_archs
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


@ti.all_archs
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
            y[i] = s.atomic_add(step)

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + step


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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
                j = s.atomic_add(s)
                k = j + s
                x[i] = k
            else:
                # If we look at the IR, this branch should be simplified, since nobody
                # is using atomic_add's result.
                x[i].atomic_add(i)
                x[i] += step

    func()

    for i in range(n):
        expect = i * 3 if i > boundary else (i + step)
        assert x[i] == expect


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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


@ti.all_archs
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
