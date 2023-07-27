import platform

import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu)
def test_pass_float_as_i32():
    @ti.kernel
    def foo(a: ti.i32):
        pass

    with pytest.raises(
        ti.TaichiRuntimeTypeError,
        match=r"Argument \(0,\) \(type=<class 'float'>\) cannot be converted into required type i32",
    ) as e:
        foo(1.2)


@test_utils.test(arch=ti.cpu)
def test_pass_float_as_ndarray():
    @ti.kernel
    def foo(a: ti.types.ndarray()):
        pass

    with pytest.raises(
        ti.TaichiRuntimeTypeError,
        match=r"Invalid type for argument a, got 1.2",
    ):
        foo(1.2)


@test_utils.test(arch=ti.cpu)
def test_random_python_class_as_ndarray():
    @ti.kernel
    def foo(a: ti.types.ndarray()):
        pass

    class Bla:
        pass

    with pytest.raises(
        ti.TaichiRuntimeTypeError,
        match=r"Invalid type for argument a, got",
    ):
        b = Bla()
        foo(b)


@test_utils.test(exclude=[ti.metal])
def test_pass_u64():
    if ti.lang.impl.current_cfg().arch == ti.vulkan and platform.system() == "Darwin":
        return

    @ti.kernel
    def foo(a: ti.u64):
        pass

    foo(2**64 - 1)


@test_utils.test()
def test_argument_redefinition():
    @ti.kernel
    def foo(a: ti.i32):
        a = 1

    with pytest.raises(ti.TaichiSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_argument_augassign():
    @ti.kernel
    def foo(a: ti.i32):
        a += 1

    with pytest.raises(ti.TaichiSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_argument_annassign():
    @ti.kernel
    def foo(a: ti.i32):
        a: ti.i32 = 1

    with pytest.raises(ti.TaichiSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_pass_struct_mismatch():
    sphere_type = ti.types.struct(center=ti.math.vec3, radius=float)
    circle_type = ti.types.struct(center=ti.math.vec2, radius=float)

    @ti.kernel
    def foo(sphere: sphere_type):
        pass

    with pytest.raises(
        ti.TaichiRuntimeTypeError,
        match=r"Argument <class 'taichi.lang.struct.Struct.* cannot be converted into required type <ti"
        r".StructType center=<taichi.lang.matrix.VectorType object at .*>, radius=f32, struct_methods={}>",
    ) as e:
        foo(circle_type(center=ti.math.vec2([1, 2]), radius=2.33))
