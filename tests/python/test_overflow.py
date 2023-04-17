import os
import platform

import pytest

import taichi as ti
from tests import test_utils

if os.name == "nt":
    pytest.skip(
        "Skipping on windows because fflush issues, " "see https://github.com/taichi-dev/taichi/issues/6179",
        allow_module_level=True,
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_no_debug(capfd):
    capfd.readouterr()

    @ti.kernel
    def foo() -> ti.i32:
        a = ti.i32(1073741824)
        b = ti.i32(1073741824)
        return a + b

    foo()
    ti.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" not in captured
    assert "return a + b" not in captured


def supports_overflow(arch):
    return arch != ti.vulkan or platform.system() != "Darwin"  # Vulkan on macOS do not have the validation layer.


add_table = [
    (ti.i8, 2**6),
    (ti.u8, 2**7),
    (ti.i16, 2**14),
    (ti.u16, 2**15),
    (ti.i32, 2**30),
    (ti.u32, 2**31),
    (ti.i64, 2**62),
    (ti.u64, 2**63),
]


@pytest.mark.parametrize("ty,num", add_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_add_overflow(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(x: ty) -> ty:
        a = ty(x)
        b = ty(num)
        return a + b

    foo(num)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" in captured
    assert "return a + b" in captured


@pytest.mark.parametrize("ty,num", add_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_add_no_overflow(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo() -> ty:
        a = ty(num)
        b = ty(num - 1)
        return a + b

    foo()
    ti.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" not in captured
    assert "return a + b" not in captured


sub_table = [
    (ti.i8, 2**6),
    (ti.i16, 2**14),
    (ti.i32, 2**30),
    (ti.i64, 2**62),
]


@pytest.mark.parametrize("ty,num", sub_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_sub_overflow_i(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num: ty) -> ty:
        a = ty(num)
        b = ty(-num)
        return a - b

    foo(num)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" in captured
    assert "return a - b" in captured


@pytest.mark.parametrize("ty,num", sub_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_sub_no_overflow_i(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num: ty) -> ty:
        a = ty(num)
        b = ty(-num + 1)
        return a - b

    foo(num)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" not in captured
    assert "return a - b" not in captured


@pytest.mark.parametrize("ty", [ti.u8, ti.u16, ti.u32, ti.u64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_sub_overflow_u(capfd, ty):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(x: ty, y: ty) -> ty:
        a = ty(x)
        b = ty(y)
        return a - b

    foo(1, 2)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" in captured
    assert "return a - b" in captured


@pytest.mark.parametrize("ty", [ti.u8, ti.u16, ti.u32, ti.u64])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_sub_no_overflow_u(capfd, ty):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(x: ty) -> ty:
        a = ty(x)
        b = ty(x)
        return a - b

    foo(1)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" not in captured
    assert "return a - b" not in captured


mul_table = [
    (ti.i8, 2**4, 2**3),
    (ti.u8, 2**4, 2**4),
    (ti.i16, 2**8, 2**7),
    (ti.u16, 2**8, 2**8),
    (ti.i32, 2**16, 2**15),
    (ti.u32, 2**16, 2**16),
    (ti.i64, 2**32, 2**31),
    (ti.u64, 2**32, 2**32),
]


@pytest.mark.parametrize("ty,num1,num2", mul_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_mul_overflow(capfd, ty, num1, num2):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    # 64-bit Multiplication overflow detection does not function correctly on old drivers.
    # See https://github.com/taichi-dev/taichi/issues/6303
    if ti.lang.impl.current_cfg().arch == ti.vulkan and id(ty) in [
        id(ti.i64),
        id(ti.u64),
    ]:
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num1: ty, num2: ty) -> ty:
        a = ty(num1 + 1)
        b = ty(num2 + 1)
        return a * b

    foo(num1, num2)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Multiplication overflow detected" in captured
    assert "return a * b" in captured


@pytest.mark.parametrize("ty,num1,num2", mul_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_mul_no_overflow(capfd, ty, num1, num2):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num1: ty, num2: ty) -> ty:
        a = ty(num1 + 1)
        b = ty(num2 - 1)
        return a * b

    foo(num1, num2)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Multiplication overflow detected" not in captured
    assert "return a * b" not in captured


shl_table = [
    (ti.i8, 6),
    (ti.u8, 7),
    (ti.i16, 14),
    (ti.u16, 15),
    (ti.i32, 30),
    (ti.u32, 31),
    (ti.i64, 62),
    (ti.u64, 63),
]


@pytest.mark.parametrize("ty,num", shl_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_shl_overflow(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num: ty) -> ty:
        a = ty(2)
        b = num
        return a << b

    foo(num)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Shift left overflow detected" in captured
    assert "return a << b" in captured


@pytest.mark.parametrize("ty,num", shl_table)
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True)
def test_shl_no_overflow(capfd, ty, num):
    if not supports_overflow(ti.lang.impl.current_cfg().arch):
        return
    capfd.readouterr()

    @ti.kernel
    def foo(num: ty) -> ty:
        a = ty(2)
        b = num - 1
        return a << b

    foo(num)
    ti.sync()
    captured = capfd.readouterr().out
    assert "Shift left overflow detected" not in captured
    assert "return a << b" not in captured
