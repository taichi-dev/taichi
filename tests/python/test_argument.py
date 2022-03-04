import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.opengl)
def test_exceed_max_eight():
    @ti.kernel
    def foo1(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32, e: ti.i32, f: ti.i32,
             g: ti.i32, h: ti.i32) -> ti.i32:
        return a + b + c + d + e + f + g + h

    assert foo1(1, 2, 3, 4, 5, 6, 7, 8) == 36

    @ti.kernel
    def foo2(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32, e: ti.i32, f: ti.i32,
             g: ti.i32, h: ti.i32, i: ti.i32) -> ti.i32:
        return a + b + c + d + e + f + g + h + i

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 8 on opengl backend."
    ):
        foo2(1, 2, 3, 4, 5, 6, 7, 8, 9)


@test_utils.test(arch=ti.cc)
def test_exceed_max_eight():
    @ti.kernel
    def foo1(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32, e: ti.i32, f: ti.i32,
             g: ti.i32, h: ti.i32) -> ti.i32:
        return a + b + c + d + e + f + g + h

    assert foo1(1, 2, 3, 4, 5, 6, 7, 8) == 36

    @ti.kernel
    def foo2(a: ti.i32, b: ti.i32, c: ti.i32, d: ti.i32, e: ti.i32, f: ti.i32,
             g: ti.i32, h: ti.i32, i: ti.i32) -> ti.i32:
        return a + b + c + d + e + f + g + h + i

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 8 on cc backend."
    ):
        foo2(1, 2, 3, 4, 5, 6, 7, 8, 9)


@test_utils.test(arch=ti.cuda)
def test_exceed_max_64():
    N = 64

    @ti.kernel
    def foo1(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)
    assert foo1(A) == 64

    N = 65

    @ti.kernel
    def foo2(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 64 on cuda backend."
    ):
        foo2(A)


@test_utils.test(arch=ti.metal)
def test_exceed_max_64():
    N = 64

    @ti.kernel
    def foo1(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)
    assert foo1(A) == 64

    N = 65

    @ti.kernel
    def foo2(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 64 on metal backend."
    ):
        foo2(A)


@test_utils.test(arch=ti.vulkan)
def test_exceed_max_64():
    N = 64

    @ti.kernel
    def foo1(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)
    assert foo1(A) == 64

    N = 65

    @ti.kernel
    def foo2(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 64 on vulkan backend."
    ):
        foo2(A)


@test_utils.test(arch=ti.x64)
def test_exceed_max_64():
    N = 64

    @ti.kernel
    def foo1(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)
    assert foo1(A) == 64

    N = 65

    @ti.kernel
    def foo2(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 64 on x64 backend."
    ):
        foo2(A)


@test_utils.test(arch=ti.arm64)
def test_exceed_max_64():
    N = 64

    @ti.kernel
    def foo1(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)
    assert foo1(A) == 64

    N = 65

    @ti.kernel
    def foo2(a: ti.types.vector(N, ti.i32)) -> ti.i32:
        return a.sum()

    A = ti.Vector([1] * N)

    with pytest.raises(
            ti.TaichiRuntimeError,
            match=
            "The number of elements in kernel arguments is too big! Do not exceed 64 on arm64 backend."
    ):
        foo2(A)
