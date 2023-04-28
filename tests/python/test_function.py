import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_function_without_return():
    x = ti.field(ti.i32, shape=())

    @ti.experimental.real_func
    def foo(val: ti.i32):
        x[None] += val

    @ti.kernel
    def run():
        foo(40)
        foo(2)

    x[None] = 0
    run()
    assert x[None] == 42


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_function_with_return():
    x = ti.field(ti.i32, shape=())

    @ti.experimental.real_func
    def foo(val: ti.i32) -> ti.i32:
        x[None] += val
        return val

    @ti.kernel
    def run():
        a = foo(40)
        foo(2)
        assert a == 40

    x[None] = 0
    run()
    assert x[None] == 42


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_call_expressions():
    x = ti.field(ti.i32, shape=())

    @ti.experimental.real_func
    def foo(val: ti.i32) -> ti.i32:
        if x[None] > 10:
            x[None] += 1
        x[None] += val
        return 0

    @ti.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0

    x[None] = 0
    run()
    assert x[None] == 26


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_default_templates():
    @ti.func
    def func1(x: ti.template()):
        x = 1

    @ti.func
    def func2(x: ti.template()):
        x += 1

    @ti.func
    def func3(x):
        x = 1

    @ti.func
    def func4(x):
        x += 1

    @ti.func
    def func1_field(x: ti.template()):
        x[None] = 1

    @ti.func
    def func2_field(x: ti.template()):
        x[None] += 1

    @ti.func
    def func3_field(x):
        x[None] = 1

    @ti.func
    def func4_field(x):
        x[None] += 1

    v = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def run_func():
        a = 0
        func1(a)
        assert a == 1
        b = 0
        func2(b)
        assert b == 1
        c = 0
        func3(c)
        assert c == 0
        d = 0
        func4(d)
        assert d == 0

        v[None] = 0
        func1_field(v)
        assert v[None] == 1
        v[None] = 0
        func2_field(v)
        assert v[None] == 1
        v[None] = 0
        func3_field(v)
        assert v[None] == 1
        v[None] = 0
        func4_field(v)
        assert v[None] == 1

    run_func()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_experimental_templates():
    x = ti.field(ti.i32, shape=())
    y = ti.field(ti.i32, shape=())
    answer = ti.field(ti.i32, shape=8)

    @ti.kernel
    def kernel_inc(x: ti.template()):
        x[None] += 1

    def run_kernel():
        x[None] = 10
        y[None] = 20
        kernel_inc(x)
        assert x[None] == 11
        assert y[None] == 20
        kernel_inc(y)
        assert x[None] == 11
        assert y[None] == 21

    @ti.experimental.real_func
    def inc(x: ti.template()):
        x[None] += 1

    @ti.kernel
    def run_func():
        x[None] = 10
        y[None] = 20
        inc(x)
        answer[0] = x[None]
        answer[1] = y[None]
        inc(y)
        answer[2] = x[None]
        answer[3] = y[None]

    def verify():
        assert answer[0] == 11
        assert answer[1] == 20
        assert answer[2] == 11
        assert answer[3] == 21

    run_kernel()
    run_func()
    verify()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_missing_arg_annotation():
    with pytest.raises(ti.TaichiSyntaxError, match="must be type annotated"):

        @ti.experimental.real_func
        def add(a, b: ti.i32) -> ti.i32:
            return a + b


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_missing_return_annotation():
    with pytest.raises(ti.TaichiCompilationError, match="return value must be annotated"):

        @ti.experimental.real_func
        def add(a: ti.i32, b: ti.i32):
            return a + b

        @ti.kernel
        def run():
            add(30, 2)

        run()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_different_argument_type():
    @ti.experimental.real_func
    def add(a: ti.f32, b: ti.f32) -> ti.f32:
        return a + b

    @ti.kernel
    def run() -> ti.i32:
        return add(1, 2)

    assert run() == 3


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda], cuda_stack_limit=8192)
def test_recursion():
    @ti.experimental.real_func
    def sum(f: ti.template(), l: ti.i32, r: ti.i32) -> ti.i32:
        if l == r:
            return f[l]
        else:
            return sum(f, l, (l + r) // 2) + sum(f, (l + r) // 2 + 1, r)

    f = ti.field(ti.i32, shape=100)
    for i in range(100):
        f[i] = i

    @ti.kernel
    def get_sum() -> ti.i32:
        return sum(f, 0, 99)

    assert get_sum() == 99 * 50


@pytest.mark.run_in_serial
@test_utils.test(arch=[ti.cpu, ti.cuda], cuda_stack_limit=32768)
def test_deep_recursion():
    @ti.experimental.real_func
    def sum_func(n: ti.i32) -> ti.i32:
        if n == 0:
            return 0
        return sum_func(n - 1) + n

    @ti.kernel
    def sum(n: ti.i32) -> ti.i32:
        return sum_func(n)

    assert sum(100) == 5050


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_multiple_return():
    x = ti.field(ti.i32, shape=())

    @ti.experimental.real_func
    def foo(val: ti.i32) -> ti.i32:
        if x[None] > 10:
            if x[None] > 20:
                return 1
            x[None] += 1
        x[None] += val
        return 0

    @ti.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0
        assert foo(100) == 1

    x[None] = 0
    run()
    assert x[None] == 26


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_return_in_for():
    @ti.experimental.real_func
    def foo() -> ti.i32:
        for i in range(10):
            return 42

    @ti.kernel
    def bar() -> ti.i32:
        return foo()

    assert bar() == 42


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_return_in_while():
    @ti.experimental.real_func
    def foo() -> ti.i32:
        i = 1
        while i:
            return 42

    @ti.kernel
    def bar() -> ti.i32:
        return foo()

    assert bar() == 42


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_return_in_if_in_for():
    @ti.experimental.real_func
    def foo(a: ti.i32) -> ti.i32:
        s = 0
        for i in range(100):
            if i == a + 1:
                return s
            s = s + i
        return s

    @ti.kernel
    def bar(a: ti.i32) -> ti.i32:
        return foo(a)

    assert bar(10) == 11 * 5
    assert bar(200) == 99 * 50


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_ref():
    @ti.experimental.real_func
    def foo(a: ti.ref(ti.f32)):
        a = 7

    @ti.kernel
    def bar():
        a = 5.0
        foo(a)
        assert a == 7

    bar()


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_ref_atomic():
    # FIXME: failed test on Pascal (and potentially older) architecture.
    # Please remove this guardiance when you fix this issue
    cur_arch = ti.lang.impl.get_runtime().prog.config().arch
    if cur_arch == ti.cuda and ti.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip(
            "Skip this test on Pascal (and potentially older) architecture, ask turbo0628/Proton for more information"
        )

    @ti.experimental.real_func
    def foo(a: ti.ref(ti.f32)):
        a += a

    @ti.kernel
    def bar():
        a = 5.0
        foo(a)
        assert a == 10.0

    bar()


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_func_ndarray_arg():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.func
    def test(a: ti.types.ndarray(ndim=1)):
        a[0] = [100, 100, 100]

    @ti.kernel
    def test_k(x: ti.types.ndarray(ndim=1)):
        test(x)

    @ti.func
    def test_error_func(a: ti.types.ndarray(dtype=ti.math.vec2, ndim=1)):
        a[0] = [100, 100]

    @ti.kernel
    def test_error(x: ti.types.ndarray(ndim=1)):
        test_error_func(x)

    arr = ti.ndarray(vec3, shape=(4))
    arr[0] = [20, 20, 20]
    test_k(arr)

    assert arr[0] == [20, 20, 20]

    with pytest.raises(ti.TaichiCompilationError, match=r"Invalid argument into ti.types.ndarray()"):
        test_error(arr)


@test_utils.test(debug=True)
def test_func_matrix_arg():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.func
    def test(a: vec3):
        a[0] = 100

    @ti.kernel
    def test_k():
        x = ti.Matrix([3, 4, 5])
        x[0] = 20
        test(x)

        assert x[0] == 20

    test_k()


@test_utils.test()
def test_func_matrix_arg_with_error():
    vec3 = ti.types.vector(3, ti.f32)

    @ti.func
    def test(a: vec3):
        a[0] = 100

    @ti.kernel
    def test_error():
        x = ti.Matrix([3, 4])
        test(x)

    with pytest.raises(ti.TaichiSyntaxError, match=r"is expected to be a Matrix with n 3, but got 2"):
        test_error()


@test_utils.test(debug=True)
def test_func_struct_arg():
    @ti.dataclass
    class C:
        i: int

    @ti.func
    def f(c: C):
        return c.i

    @ti.kernel
    def k():
        c = C(i=2)
        assert f(c) == 2

    k()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_matrix_arg():
    @ti.experimental.real_func
    def mat_arg(a: ti.math.mat2, b: ti.math.vec2) -> float:
        return a[0, 0] + a[0, 1] + a[1, 0] + a[1, 1] + b[0] + b[1]

    b = ti.Vector.field(n=2, dtype=float, shape=())
    b[()][0] = 5
    b[()][1] = 6

    @ti.kernel
    def foo() -> float:
        a = ti.math.mat2(1, 2, 3, 4)
        return mat_arg(a, b[()])

    assert foo() == pytest.approx(21)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_struct_ret():
    s = ti.types.struct(a=ti.i16, b=ti.f64)

    @ti.experimental.real_func
    def bar() -> s:
        return s(a=123, b=ti.f64(1.2345e300))

    @ti.kernel
    def foo() -> ti.f64:
        a = bar()
        return a.a * a.b

    assert foo() == pytest.approx(123 * 1.2345e300)


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_struct_ret_with_matrix():
    s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)
    s1 = ti.types.struct(a=ti.f32, b=s0)

    @ti.experimental.real_func
    def bar() -> s1:
        return s1(a=1, b=s0(a=ti.Vector([100, 0.2, 3], dt=ti.f32), b=65537))

    @ti.kernel
    def foo() -> ti.f32:
        s = bar()
        return s.a + s.b.a[0] + s.b.a[1] + s.b.a[2] + s.b.b

    assert foo() == pytest.approx(105.2)
