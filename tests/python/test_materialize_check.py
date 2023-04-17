import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_check_field_not_placed():
    a = ti.field(ti.i32)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(RuntimeError, match=r"These field\(s\) are not placed.*"):
        foo()


@test_utils.test()
def test_check_grad_field_not_placed():
    a = ti.field(ti.f32, needs_grad=True)
    ti.root.dense(ti.i, 1).place(a)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_vector_field_not_placed():
    b = ti.Vector.field(3, ti.f32, needs_grad=True)
    ti.root.dense(ti.i, 1).place(b)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_matrix_field_not_placed():
    c = ti.Matrix.field(2, 3, ti.f32, needs_grad=True)
    ti.root.dense(ti.i, 1).place(c)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_grad_struct_field_not_placed():
    d = ti.Struct.field(
        {
            "pos": ti.types.vector(3, float),
            "vel": ti.types.vector(3, float),
            "acc": ti.types.vector(3, float),
            "mass": ti.f32,
        },
        needs_grad=True,
    )
    ti.root.dense(ti.i, 1).place(d)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_grad=True`, however their grad field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_field_not_placed():
    a = ti.field(ti.f32, needs_dual=True)
    ti.root.dense(ti.i, 1).place(a)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_vector_field_not_placed():
    b = ti.Vector.field(3, ti.f32, needs_dual=True)
    ti.root.dense(ti.i, 1).place(b)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_matrix_field_not_placed():
    c = ti.Matrix.field(2, 3, ti.f32, needs_dual=True)
    ti.root.dense(ti.i, 1).place(c)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_dual_struct_field_not_placed():
    d = ti.Struct.field(
        {
            "pos": ti.types.vector(3, float),
            "vel": ti.types.vector(3, float),
            "acc": ti.types.vector(3, float),
            "mass": ti.f32,
        },
        needs_dual=True,
    )
    ti.root.dense(ti.i, 1).place(d)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
        RuntimeError,
        match=r"These field\(s\) requrie `needs_dual=True`, however their dual field\(s\) are not placed.*",
    ):
        foo()


@test_utils.test()
def test_check_matrix_field_member_shape():
    a = ti.Matrix.field(2, 2, ti.i32)
    ti.root.dense(ti.i, 10).place(a.get_scalar_field(0, 0))
    ti.root.dense(ti.i, 11).place(a.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 10).place(a.get_scalar_field(1, 0))
    ti.root.dense(ti.i, 11).place(a.get_scalar_field(1, 1))

    @ti.kernel
    def foo():
        pass

    with pytest.raises(RuntimeError, match=r"Members of the following field have different shapes.*"):
        foo()
