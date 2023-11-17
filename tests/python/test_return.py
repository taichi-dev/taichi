import sys

import pytest
from typing import Tuple
from pytest import approx

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_return_without_type_hint():
    @ti.kernel
    def kernel():
        return 1

    with pytest.raises(ti.TaichiCompilationError):
        kernel()


def test_const_func_ret():
    ti.init()

    @ti.kernel
    def func1() -> ti.f32:
        return 3

    @ti.kernel
    def func2() -> ti.i32:
        return 3.3  # return type mismatch, will be auto-casted into ti.i32

    assert func1() == test_utils.approx(3)
    assert func2() == 3


@pytest.mark.parametrize(
    "dt1,dt2,dt3,castor",
    [
        (ti.i32, ti.f32, ti.f32, float),
        (ti.f32, ti.i32, ti.f32, float),
        (ti.i32, ti.f32, ti.i32, int),
        (ti.f32, ti.i32, ti.i32, int),
    ],
)
@test_utils.test()
def test_binary_func_ret(dt1, dt2, dt3, castor):
    @ti.kernel
    def func(a: dt1, b: dt2) -> dt3:
        return a * b

    if ti.types.is_integral(dt1):
        xs = list(range(4))
    else:
        xs = [0.2, 0.4, 0.8, 1.0]

    if ti.types.is_integral(dt2):
        ys = list(range(4))
    else:
        ys = [0.2, 0.4, 0.8, 1.0]

    for x, y in zip(xs, ys):
        assert func(x, y) == test_utils.approx(castor(x * y))


@test_utils.test()
def test_return_in_static_if():
    @ti.kernel
    def foo(a: ti.template()) -> ti.i32:
        if ti.static(a == 1):
            return 1
        elif ti.static(a == 2):
            return 2
        return 3

    assert foo(1) == 1
    assert foo(2) == 2
    assert foo(123) == 3


@test_utils.test()
def test_func_multiple_return():
    @ti.func
    def safe_sqrt(a):
        if a > 0:
            return ti.sqrt(a)
        else:
            return 0.0

    @ti.kernel
    def kern(a: float):
        print(safe_sqrt(a))

    with pytest.raises(
        ti.TaichiCompilationError,
        match="Return inside non-static if/for is not supported",
    ):
        kern(-233)


@test_utils.test()
def test_return_inside_static_for():
    @ti.kernel
    def foo() -> ti.i32:
        a = 0
        for i in ti.static(range(10)):
            a += i * i
            if ti.static(i == 8):
                return a

    assert foo() == 204


@test_utils.test()
def test_return_inside_non_static_for():
    with pytest.raises(
        ti.TaichiCompilationError,
        match="Return inside non-static if/for is not supported",
    ):

        @ti.kernel
        def foo() -> ti.i32:
            for i in range(10):
                return i

        foo()


@test_utils.test()
def test_kernel_no_return():
    with pytest.raises(
        ti.TaichiSyntaxError,
        match="Kernel has a return type but does not have a return statement",
    ):

        @ti.kernel
        def foo() -> ti.i32:
            pass

        foo()


@test_utils.test()
def test_func_no_return():
    with pytest.raises(
        ti.TaichiCompilationError,
        match="Function has a return type but does not have a return statement",
    ):

        @ti.func
        def bar() -> ti.i32:
            pass

        @ti.kernel
        def foo() -> ti.i32:
            return bar()

        foo()


@test_utils.test()
def test_void_return():
    @ti.kernel
    def foo():
        return

    foo()


@test_utils.test()
def test_return_none():
    @ti.kernel
    def foo():
        return None

    foo()


@test_utils.test(exclude=[ti.metal, ti.vulkan, ti.gles])
def test_return_uint64():
    @ti.kernel
    def foo() -> ti.u64:
        return ti.u64(2**64 - 1)

    assert foo() == 2**64 - 1


@test_utils.test(exclude=[ti.metal, ti.vulkan, ti.gles])
def test_return_uint64_vec():
    @ti.kernel
    def foo() -> ti.types.vector(2, ti.u64):
        return ti.Vector([ti.u64(2**64 - 1), ti.u64(2**64 - 1)])

    assert foo()[0] == 2**64 - 1


@test_utils.test()
def test_struct_ret_with_matrix():
    s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)
    s1 = ti.types.struct(a=ti.f32, b=s0)

    @ti.kernel
    def foo() -> s1:
        return s1(a=1, b=s0(a=ti.math.vec3([100, 0.2, 3]), b=65537))

    ret = foo()
    assert ret.a == approx(1)
    assert ret.b.a[0] == approx(100)
    assert ret.b.a[1] == approx(0.2)
    assert ret.b.a[2] == approx(3)
    assert ret.b.b == 1


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_tuple_ret_39():
    s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)

    @ti.real_func
    def foo() -> tuple[ti.f32, s0]:
        return 1, s0(a=ti.math.vec3([100, 0.2, 3]), b=65537)

    @ti.kernel
    def bar() -> tuple[ti.f32, s0]:
        return foo()

    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_real_func_tuple_ret_typing_tuple():
    s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)

    @ti.real_func
    def foo() -> Tuple[ti.f32, s0]:
        return 1, s0(a=ti.math.vec3([100, 0.2, 3]), b=65537)

    @ti.kernel
    def bar() -> Tuple[ti.f32, s0]:
        return foo()

    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test(arch=[ti.cpu, ti.cuda], debug=True)
def test_real_func_tuple_ret():
    s0 = ti.types.struct(a=ti.math.vec3, b=ti.i16)

    @ti.real_func
    def foo() -> (ti.f32, s0):
        return 1, s0(a=ti.math.vec3([100, 0.2, 3]), b=65537)

    @ti.kernel
    def bar() -> (ti.f32, s0):
        return foo()

    # bar()
    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test()
def test_return_type_mismatch_1():
    with pytest.raises(ti.TaichiCompilationError):

        @ti.kernel
        def foo() -> ti.i32:
            return ti.math.vec3([1, 2, 3])

        foo()


@test_utils.test()
def test_return_type_mismatch_2():
    with pytest.raises(ti.TaichiCompilationError):

        @ti.kernel
        def foo() -> ti.math.vec4:
            return ti.math.vec3([1, 2, 3])

        foo()


@test_utils.test()
def test_return_type_mismatch_3():
    sphere_type = ti.types.struct(center=ti.math.vec3, radius=float)
    circle_type = ti.types.struct(center=ti.math.vec2, radius=float)
    sphere_type_ = ti.types.struct(center=ti.math.vec3, radius=int)

    @ti.kernel
    def foo() -> sphere_type:
        return circle_type(center=ti.math.vec2([1, 2]), radius=2)

    @ti.kernel
    def bar() -> sphere_type:
        return sphere_type_(center=ti.math.vec3([1, 2, 3]), radius=2)

    with pytest.raises(ti.TaichiCompilationError):
        foo()

    with pytest.raises(ti.TaichiCompilationError):
        bar()


@test_utils.test()
def test_func_scalar_return_cast():
    @ti.func
    def bar(a: ti.f32) -> ti.i32:
        return a

    @ti.kernel
    def foo(a: ti.f32) -> ti.f32:
        return bar(a)

    assert foo(1.5) == 1.0


@test_utils.test()
def test_return_struct_field():
    tp = ti.types.struct(a=ti.i32)

    f = tp.field(shape=1)

    @ti.func
    def bar() -> tp:
        return f[0]

    @ti.kernel
    def foo() -> tp:
        return bar()

    assert foo().a == 0


@test_utils.test(exclude=[ti.amdgpu])
def test_ret_4k():
    vec1024 = ti.types.vector(1024, ti.i32)

    @ti.kernel
    def foo() -> vec1024:
        ret = vec1024(0)
        for i in range(1024):
            ret[i] = i
        return ret

    ret = foo()
    for i in range(1024):
        assert ret[i] == i
