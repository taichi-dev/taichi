import numpy as np
import pytest
from taichi._testing import approx
from taichi.lang import impl
from taichi.lang.util import has_pytorch

import taichi as ti


@ti.test()
def test_binop():
    @ti.kernel
    def foo(x: ti.i32, y: ti.i32, a: ti.template()):
        a[0] = x + y
        a[1] = x - y
        a[2] = x * y
        a[3] = impl.ti_float(x) / y
        a[4] = x // y
        a[5] = x % y
        a[6] = x**y
        a[7] = x << y
        a[8] = x >> y
        a[9] = x | y
        a[10] = x ^ y
        a[11] = x & y

    x = 37
    y = 3
    a = ti.field(ti.f32, shape=(12, ))
    b = ti.field(ti.f32, shape=(12, ))

    a[0] = x + y
    a[1] = x - y
    a[2] = x * y
    a[3] = x / y
    a[4] = x // y
    a[5] = x % y
    a[6] = x**y
    a[7] = x << y
    a[8] = x >> y
    a[9] = x | y
    a[10] = x ^ y
    a[11] = x & y

    foo(x, y, b)

    for i in range(12):
        assert a[i] == approx(b[i])


@ti.test()
def test_augassign():
    @ti.kernel
    def foo(x: ti.i32, y: ti.i32, a: ti.template(), b: ti.template()):
        for i in a:
            a[i] = x
        a[0] += y
        a[1] -= y
        a[2] *= y
        a[3] //= y
        a[4] %= y
        a[5] **= y
        a[6] <<= y
        a[7] >>= y
        a[8] |= y
        a[9] ^= y
        a[10] &= y
        b[0] = x
        b[0] /= y

    x = 37
    y = 3
    a = ti.field(ti.i32, shape=(11, ))
    b = ti.field(ti.i32, shape=(11, ))
    c = ti.field(ti.f32, shape=(1, ))
    d = ti.field(ti.f32, shape=(1, ))

    a[0] = x + y
    a[1] = x - y
    a[2] = x * y
    a[3] = x // y
    a[4] = x % y
    a[5] = x**y
    a[6] = x << y
    a[7] = x >> y
    a[8] = x | y
    a[9] = x ^ y
    a[10] = x & y
    c[0] = x / y

    foo(x, y, b, d)

    for i in range(11):
        assert a[i] == b[i]
    assert c[0] == approx(d[0])


@ti.test()
def test_unaryop():
    @ti.kernel
    def foo(x: ti.i32, a: ti.template()):
        a[0] = +x
        a[1] = -x
        a[2] = not x
        a[3] = ~x

    x = 1234
    a = ti.field(ti.i32, shape=(4, ))
    b = ti.field(ti.i32, shape=(4, ))

    a[0] = +x
    a[1] = -x
    a[2] = not x
    a[3] = ~x

    foo(x, b)

    for i in range(4):
        assert a[i] == b[i]


@ti.test()
def test_boolop():
    @ti.kernel
    def foo(a: ti.template()):
        a[0] = 0 and 0
        a[1] = 0 and 1
        a[2] = 1 and 0
        a[3] = 1 and 1
        a[4] = 0 or 0
        a[5] = 0 or 1
        a[6] = 1 or 0
        a[7] = 1 or 1
        a[8] = 1 and 1 and 1 and 1
        a[9] = 1 and 1 and 1 and 0
        a[10] = 0 or 0 or 0 or 0
        a[11] = 0 or 0 or 1 or 0

    a = ti.field(ti.i32, shape=(12, ))
    b = ti.field(ti.i32, shape=(12, ))

    a[0] = 0 and 0
    a[1] = 0 and 1
    a[2] = 1 and 0
    a[3] = 1 and 1
    a[4] = 0 or 0
    a[5] = 0 or 1
    a[6] = 1 or 0
    a[7] = 1 or 1
    a[8] = 1 and 1 and 1 and 1
    a[9] = 1 and 1 and 1 and 0
    a[10] = 0 or 0 or 0 or 0
    a[11] = 0 or 0 or 1 or 0

    foo(b)

    for i in range(12):
        assert a[i] == b[i]


@ti.test()
def test_compare_fail():
    with pytest.raises(ti.TaichiCompilationError,
                       match='"Is" is not supported in Taichi kernels.'):

        @ti.kernel
        def foo():
            1 is [1]

        foo()


@ti.test()
def test_single_compare():
    @ti.kernel
    def foo(a: ti.template(), b: ti.template(), c: ti.template()):
        for i in ti.static(range(3)):
            c[i * 6] = a[i] == b[i]
            c[i * 6 + 1] = a[i] != b[i]
            c[i * 6 + 2] = a[i] < b[i]
            c[i * 6 + 3] = a[i] <= b[i]
            c[i * 6 + 4] = a[i] > b[i]
            c[i * 6 + 5] = a[i] >= b[i]

    a = ti.Vector([1, 1, 2])
    b = ti.Vector([2, 1, 1])
    c = ti.field(ti.i32, shape=(18, ))
    d = ti.field(ti.i32, shape=(18, ))

    for i in range(3):
        c[i * 6] = a[i] == b[i]
        c[i * 6 + 1] = a[i] != b[i]
        c[i * 6 + 2] = a[i] < b[i]
        c[i * 6 + 3] = a[i] <= b[i]
        c[i * 6 + 4] = a[i] > b[i]
        c[i * 6 + 5] = a[i] >= b[i]

    foo(a, b, d)
    for i in range(18):
        assert c[i] == d[i]


@ti.test()
def test_chain_compare():
    @ti.kernel
    def foo(a: ti.i32, b: ti.i32, c: ti.template()):
        c[0] = a == b == a
        c[1] = a == b != a
        c[2] = a != b == a
        c[3] = a < b > a
        c[4] = a > b < a
        c[5] = a < b < a
        c[6] = a > b > a
        c[7] = a == a == a == a
        c[8] = a == a == a != a
        c[9] = a < b > a < b
        c[10] = a > b > a < b

    a = 1
    b = 2
    c = ti.field(ti.i32, shape=(11, ))
    d = ti.field(ti.i32, shape=(11, ))

    c[0] = a == b == a
    c[1] = a == b != a
    c[2] = a != b == a
    c[3] = a < b > a
    c[4] = a > b < a
    c[5] = a < b < a
    c[6] = a > b > a
    c[7] = a == a == a == a
    c[8] = a == a == a != a
    c[9] = a < b > a < b
    c[10] = a > b > a < b

    foo(a, b, d)
    for i in range(11):
        assert c[i] == d[i]


@ti.test()
def test_return():
    @ti.kernel
    def foo(x: ti.i32) -> ti.i32:
        return x + 1

    assert foo(1) == 2


@ti.test()
def test_format_print():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo():
        a[0] = 1.0
        a[5] = 2.0
        print('Test if the string.format and fstring print works')
        print('string.format: a[0]={}, a[5]={}'.format(a[0], a[5]))
        print(f'fstring: a[0]={a[0]}, a[5]={a[5]}')


@ti.test(print_preprocessed_ir=True)
def test_if():
    @ti.kernel
    def foo(x: ti.i32) -> ti.i32:
        ret = 0
        if x:
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@ti.test(print_preprocessed_ir=True)
def test_static_if():
    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        ret = 0
        if ti.static(x):
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@ti.test(print_preprocessed_ir=True)
def test_struct_for():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in a:
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@ti.test(print_preprocessed_ir=True)
def test_grouped_struct_for():
    a = ti.field(ti.i32, shape=(4, 4))

    @ti.kernel
    def foo(x: ti.i32):
        for I in ti.grouped(a):
            a[I] = x

    x = 5
    foo(x)
    for i in range(4):
        for j in range(4):
            assert a[i, j] == 5


@ti.test(print_preprocessed_ir=True)
def test_static_for():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in ti.static(range(10)):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@ti.test(print_preprocessed_ir=True)
def test_static_grouped_for():
    a = ti.field(ti.i32, shape=(4, 4))

    @ti.kernel
    def foo(x: ti.i32):
        for i in ti.static(ti.grouped(ti.ndrange((1, 3), (1, 3)))):
            a[i] = x

    x = 5
    foo(x)
    for i in range(4):
        for j in range(4):
            if 1 <= i < 3 and 1 <= j < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 0


@ti.test(print_preprocessed_ir=True)
def test_range_for_single_argument():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in range(5):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        if i < 5:
            assert a[i] == 5
        else:
            assert a[i] == 0


@ti.test(print_preprocessed_ir=True)
def test_range_for_two_arguments():
    a = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo(x: ti.i32):
        for i in range(3, 7):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        if 3 <= i < 7:
            assert a[i] == 5
        else:
            assert a[i] == 0


@ti.test()
def test_range_for_three_arguments():
    a = ti.field(ti.i32, shape=(10, ))

    with pytest.raises(ti.TaichiCompilationError,
                       match='Range should have 1 or 2 arguments, found 3'):

        @ti.kernel
        def foo(x: ti.i32):
            for i in range(3, 7, 2):
                a[i] = x

        x = 5
        foo(x)


@ti.test(print_preprocessed_ir=True)
def test_ndrange_for():
    x = ti.field(ti.f32, shape=(16, 32, 64))

    @ti.kernel
    def func():
        for i, j, k in ti.ndrange((4, 10), (3, 8), 17):
            x[i, j, k] = i + j * 10 + k * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 10 and 3 <= j < 8 and k < 17:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@ti.test(print_preprocessed_ir=True)
def test_grouped_ndrange_for():
    x = ti.field(ti.i32, shape=(6, 6, 6))
    y = ti.field(ti.i32, shape=(6, 6, 6))

    @ti.kernel
    def func():
        lower = ti.Vector([0, 1, 2])
        upper = ti.Vector([3, 4, 5])
        for I in ti.grouped(
                ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]),
                           (lower[2], upper[2]))):
            x[I] = I[0] + I[1] + I[2]
        for i in range(0, 3):
            for j in range(1, 4):
                for k in range(2, 5):
                    y[i, j, k] = i + j + k

    func()

    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert x[i, j, k] == y[i, j, k]


@ti.test(print_preprocessed_ir=True)
def test_static_for_break():
    n = 10

    @ti.kernel
    def foo(a: ti.template()):
        for i in ti.static(range(n)):
            a[i] = 3
            if ti.static(i >= 5):
                break
                a[i] = 10
            a[i] = 5

    a = ti.field(ti.i32, shape=(n, ))
    foo(a)
    for i in range(n):
        if i < 5:
            assert a[i] == 5
        elif i == 5:
            assert a[i] == 3
        else:
            assert a[i] == 0


@ti.test(print_preprocessed_ir=True)
def test_static_grouped_for_break():
    n = 4

    @ti.kernel
    def foo(a: ti.template()):
        for I in ti.static(ti.grouped(ti.ndrange(n, n))):
            a[I] = 3
            if ti.static(I[0] >= 3):
                break
                a[I] = 10
            a[I] = 5

    a = ti.field(ti.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            elif i == 3 and j == 0:
                assert a[i, j] == 3
            else:
                assert a[i, j] == 0


@ti.test(print_preprocessed_ir=True)
def test_static_for_continue():
    n = 10

    @ti.kernel
    def foo(a: ti.template()):
        for i in ti.static(range(n)):
            a[i] = 3
            if ti.static(i >= 5):
                continue
                a[i] = 10
            a[i] = 5

    a = ti.field(ti.i32, shape=(n, ))
    foo(a)
    for i in range(n):
        if i < 5:
            assert a[i] == 5
        else:
            assert a[i] == 3


@ti.test(print_preprocessed_ir=True)
def test_static_grouped_for_continue():
    n = 4

    @ti.kernel
    def foo(a: ti.template()):
        for I in ti.static(ti.grouped(ti.ndrange(n, n))):
            a[I] = 3
            if ti.static(I[0] >= 3):
                continue
                a[I] = 10
            a[I] = 5

    a = ti.field(ti.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 3


@ti.test(print_preprocessed_ir=True)
def test_for_break():
    n = 4

    @ti.kernel
    def foo(a: ti.template()):
        for i in range(n):
            for j in range(n):
                a[i, j] = 3
                if i >= 3:
                    break
                    a[i, j] = 10
                a[i, j] = 5

    a = ti.field(ti.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            elif i == 3 and j == 0:
                assert a[i, j] == 3
            else:
                assert a[i, j] == 0


@ti.test(print_preprocessed_ir=True)
def test_for_continue():
    n = 4

    @ti.kernel
    def foo(a: ti.template()):
        for i in range(n):
            for j in range(n):
                a[i, j] = 3
                if i >= 3:
                    continue
                    a[i, j] = 10
                a[i, j] = 5

    a = ti.field(ti.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 3


@ti.test()
def test_while():
    x = ti.field(ti.f32)

    N = 1

    ti.root.dense(ti.i, N).place(x)

    @ti.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            s += i
            i += 1
        x[0] = s

    func()
    assert x[0] == 45


@ti.test()
def test_while_break():
    ret = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        i = 0
        s = 0
        while True:
            s += i
            i += 1
            if i > 10:
                break
        ret[None] = s

    func()
    assert ret[None] == 55


@ti.test()
def test_while_continue():
    ret = ti.field(ti.i32, shape=())

    @ti.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            i += 1
            if i % 2 == 0:
                continue
            s += i
        ret[None] = s

    func()
    assert ret[None] == 25


@ti.test(print_preprocessed_ir=True)
def test_func():
    @ti.func
    def bar(x):
        return x * x, -x

    a = ti.field(ti.i32, shape=(10, ))
    b = ti.field(ti.i32, shape=(10, ))

    @ti.kernel
    def foo():
        for i in a:
            a[i], b[i] = bar(i)

    foo()
    for i in range(10):
        assert a[i] == i * i
        assert b[i] == -i


@ti.test(print_preprocessed_ir=True)
def test_func_in_python_func():
    @ti.func
    def bar(x: ti.template()):
        if ti.static(x):
            mat = bar(x // 2)
            mat = mat @ mat
            if ti.static(x % 2):
                mat = mat @ ti.Matrix([[1, 1], [1, 0]])
            return mat
        else:
            return ti.Matrix([[1, 0], [0, 1]])

    def fibonacci(x):
        return impl.subscript(bar(x), 1, 0)

    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        return fibonacci(x)

    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(10):
        assert foo(i) == fib[i]


@ti.test(print_preprocessed_ir=True)
def test_ifexp():
    @ti.kernel
    def foo(x: ti.i32) -> ti.i32:
        return 1 if x else 0

    assert foo(1) == 1
    assert foo(0) == 0


@ti.test(print_preprocessed_ir=True)
def test_static_ifexp():
    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        return 1 if ti.static(x) else 0

    assert foo(1) == 1
    assert foo(0) == 0


@ti.test()
def test_static_assign():
    a = ti.field(ti.i32, shape=(1, ))
    b = ti.field(ti.i32, shape=(1, ))

    @ti.kernel
    def foo(xx: ti.template(), yy: ti.template()) -> ti.i32:
        x, y = ti.static(xx, yy)
        x[0] -= 1
        y[0] -= 1
        return x[0] + y[0]

    a[0] = 2
    b[0] = 3
    assert foo(a, b) == 3


@ti.test()
def test_static_assign_element():
    with pytest.raises(
            ti.TaichiCompilationError,
            match='Static assign cannot be used on elements in arrays'):

        @ti.kernel
        def foo():
            a = ti.static([1, 2, 3])
            a[0] = ti.static(2)

        foo()


@ti.test()
def test_recreate_variable():
    with pytest.raises(ti.TaichiCompilationError,
                       match='Recreating variables is not allowed'):

        @ti.kernel
        def foo():
            a = 1
            a = ti.static(2)

        foo()


@ti.test()
def test_taichi_other_than_ti():
    import taichi as tc

    @tc.func
    def bar(x: tc.template()):
        if tc.static(x):
            mat = bar(x // 2)
            mat = mat @ mat
            if tc.static(x % 2):
                mat = mat @ tc.Matrix([[1, 1], [1, 0]])
            return mat
        else:
            return tc.Matrix([[1, 0], [0, 1]])

    def fibonacci(x):
        return impl.subscript(bar(x), 1, 0)

    @tc.kernel
    def foo(x: tc.template()) -> tc.i32:
        return fibonacci(x)

    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(10):
        assert foo(i) == fib[i]


@ti.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message():
    @ti.kernel
    def func():
        x = 20
        assert 10 <= x < 20, 'Foo bar'

    with pytest.raises(RuntimeError, match='Foo bar'):
        func()


@ti.test(require=ti.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = ti.field(dtype=int, shape=16)
    x[10] = 42

    @ti.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, 'x[%d] expect=%d got=%d' % (i, 0, x[i])

    @ti.kernel
    def assert_float():
        y = 0.5
        assert y < 0, 'y = %f' % y

    with pytest.raises(RuntimeError, match=r'x\[10\] expect=0 got=42'):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(RuntimeError, match=r'y = 0.5'):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@ti.test()
def test_dict():
    @ti.kernel
    def foo(x: ti.template()) -> ti.i32:
        a = {1: 2, 3: 4}
        b = {5: 6, **a}
        return b[x]

    assert foo(1) == 2
    with pytest.raises(ti.TaichiCompilationError):
        foo(2)


@ti.test()
def test_listcomp():
    @ti.func
    def identity(dt, n: ti.template()):
        return ti.Matrix([[ti.cast(int(i == j), dt) for j in range(n)]
                          for i in range(n)])

    @ti.kernel
    def foo(n: ti.template()) -> ti.i32:
        a = identity(ti.i32, n)
        b = [j for i in a for j in i]
        ret = 0
        for i in ti.static(range(n)):
            for j in ti.static(range(n)):
                ret += i * j * b[i * n + j]
        return ret

    assert foo(5) == 1 + 4 + 9 + 16


@ti.test()
def test_dictcomp():
    @ti.kernel
    def foo(n: ti.template()) -> ti.i32:
        a = {i: i * i for i in range(n) if i % 3 if i % 2}
        ret = 0
        for i in ti.static(range(n)):
            if ti.static(i % 3):
                if ti.static(i % 2):
                    ret += a[i]
        return ret

    assert foo(10) == 1 * 1 + 5 * 5 + 7 * 7


@ti.test()
def test_dictcomp_fail():
    @ti.kernel
    def foo(n: ti.template(), m: ti.template()) -> ti.i32:
        a = {i: i * i for i in range(n) if i % 3 if i % 2}
        return a[m]

    with pytest.raises(ti.TaichiCompilationError):
        foo(5, 2)

    with pytest.raises(ti.TaichiCompilationError):
        foo(5, 3)


@pytest.mark.skipif(not has_pytorch(), reason='Pytorch not installed.')
@ti.test(arch=[ti.cpu, ti.cuda, ti.opengl])
def test_ndarray():
    n = 4
    m = 7

    @ti.kernel
    def run(x: ti.any_arr(element_dim=2, layout=ti.Layout.AOS),
            y: ti.any_arr()):
        for i in ti.static(range(n)):
            for j in ti.static(range(m)):
                x[i, j][0, 0] += i + j + y[i, j]

    a = ti.Matrix.ndarray(1, 1, ti.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j][0, 0] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j][0, 0] == i * j + i + j + 1


@ti.test(arch=ti.cpu)
def test_sparse_matrix_builder():
    n = 8
    Abuilder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @ti.kernel
    def fill(Abuilder: ti.linalg.sparse_matrix_builder()):
        for i, j in ti.static(ti.ndrange(n, n)):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@ti.test()
def test_func_default_value():
    @ti.func
    def bar(s, t=1):
        return s + t

    @ti.kernel
    def foo() -> ti.i32:
        return bar(1)

    assert foo() == 2


@ti.test()
def test_func_default_value_fail():
    with pytest.raises(ti.TaichiCompilationError):

        @ti.func
        def bar(s, t=1):
            return s + t

        @ti.kernel
        def foo() -> ti.i32:
            return bar(1, 2, 3)

        foo()


@ti.test()
def test_raise():
    dim = 1
    m = ti.Matrix.field(dim, dim, ti.f32)
    ti.root.place(m)

    with pytest.raises(
            ti.TaichiCompilationError,
            match="Polar decomposition only supports 2D and 3D matrices."):

        @ti.kernel
        def foo():
            ti.polar_decompose(m, ti.f32)

        foo()


@ti.test()
def test_scalar_argument():
    @ti.kernel
    def add(a: ti.f32, b: ti.f32) -> ti.f32:
        a = a + b
        return a

    assert add(1.0, 2.0) == approx(3.0)


@ti.test()
def test_default_template_args_on_func():
    @ti.func
    def bar(a: ti.template() = 123):
        return a

    @ti.kernel
    def foo() -> ti.i32:
        return bar()

    assert foo() == 123


@ti.test()
def test_grouped_static_for_cast():
    @ti.kernel
    def foo() -> ti.f32:
        ret = 0.
        for I in ti.static(ti.grouped(ti.ndrange((4, 5), (3, 5), 5))):
            tmp = I.cast(float)
            ret += tmp[2] / 2
        return ret

    assert foo() == approx(10)
