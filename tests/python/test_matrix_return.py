import os

import pytest

import taichi as ti

### `ti.test`


@test_utils.test(arch=ti.cpu)
def test_arch_cpu():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.gpu)
def test_arch_gpu():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.cuda)
def test_arch_cuda():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.vulkan)
def test_arch_vulkan():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.cc)
def test_arch_cc():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.metal)
def test_arch_metal():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6


@test_utils.test(arch=ti.opengl)
def test_arch_opengl():
    @ti.kernel
    def func() -> ti.types.matrix(2, 3, ti.i32):
        return ti.Matrix([[1, 2, 3], [4, 5, 6]])

    assert func()[1, 2] == 6
