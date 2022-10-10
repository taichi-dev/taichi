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
            match=
            r"Argument 0 \(type=<class 'float'>\) cannot be converted into required type i32"
    ) as e:
        foo(1.2)


@test_utils.test(exclude=[ti.metal])
def test_pass_u64():
    @ti.kernel
    def foo(a: ti.u64):
        pass

    foo(2**64 - 1)


@test_utils.test()
def test_argument_redefinition():
    @ti.kernel
    def foo(a: ti.i32):
        a = 1

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="Kernel argument \"a\" is immutable in the kernel") as e:
        foo(5)


@test_utils.test()
def test_argument_augassign():
    @ti.kernel
    def foo(a: ti.i32):
        a += 1

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="Kernel argument \"a\" is immutable in the kernel") as e:
        foo(5)


@test_utils.test()
def test_argument_annassign():
    @ti.kernel
    def foo(a: ti.i32):
        a: ti.i32 = 1

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="Kernel argument \"a\" is immutable in the kernel") as e:
        foo(5)
