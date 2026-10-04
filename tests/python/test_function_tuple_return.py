from typing import Tuple

import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@pytest.mark.parametrize("return_type", [(ti.i32, ti.f32), Tuple[ti.i32, ti.f32], tuple[ti.i32, ti.f32]])
@test_utils.test()
def test_annotated_scalar_tuple_casts(decorator, return_type):
    @decorator
    def values() -> return_type:
        return 3.75, -2

    @ti.kernel
    def first() -> ti.f32:
        integer, real = values()
        return integer + real * 0.25

    @ti.kernel
    def second() -> ti.i32:
        integer, real = values()
        return integer * 10 + ti.cast(real, ti.i32)

    assert first() == pytest.approx(2.5)
    assert second() == 28


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_scalar_vector_tuple(decorator):
    @decorator
    def values() -> tuple[ti.i32, ti.math.vec3]:
        return 7.75, ti.Vector([1.0, 2.0, 3.0])

    @ti.kernel
    def result() -> ti.math.vec3:
        scalar, vector = values()
        return vector * scalar

    assert result().to_list() == pytest.approx([7.0, 14.0, 21.0])


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_vector_scalar_tuple(decorator):
    @decorator
    def values() -> Tuple[ti.math.vec2, ti.i32]:
        return ti.Vector([2.0, 4.0]), 3.75

    @ti.kernel
    def result() -> ti.math.vec2:
        vector, scalar = values()
        return vector + scalar

    assert result().to_list() == pytest.approx([5.0, 7.0])


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_scalar_matrix_struct_tuple(decorator):
    matrix_type = ti.types.matrix(2, 2, ti.f32)
    struct_type = ti.types.struct(value=ti.f32, offset=ti.i32)
    output = ti.Matrix.field(2, 2, dtype=ti.f32, shape=())
    scalar_output = ti.field(ti.i32, shape=())
    struct_output = struct_type.field(shape=())

    @decorator
    def values() -> (ti.i32, matrix_type, struct_type):
        return 2.75, ti.Matrix([[1.0, 2.0], [3.0, 4.0]]), struct_type(value=5.5, offset=6)

    @ti.kernel
    def result():
        scalar, matrix, structure = values()
        scalar_output[None] = scalar
        output[None] = matrix * scalar
        struct_output[None] = structure

    result()
    assert scalar_output[None] == 2
    assert output[None].to_list() == [[2.0, 4.0], [6.0, 8.0]]
    assert struct_output[None].value == pytest.approx(5.5)
    assert struct_output[None].offset == 6


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_forwarded_tuple(decorator):
    @ti.func
    def unannotated_values():
        return 1.75, -2.75

    @decorator
    def values() -> tuple[ti.i32, ti.i32]:
        return unannotated_values()

    @ti.kernel
    def result() -> ti.i32:
        first, second = values()
        return first * 10 + second

    assert result() == 8


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_vector_tuple_without_scalar_casts(decorator):
    @decorator
    def values() -> tuple[ti.math.vec2, ti.math.vec3]:
        return ti.Vector([1.0, 2.0]), ti.Vector([3.0, 4.0, 5.0])

    @ti.kernel
    def result() -> ti.f32:
        returned = values()
        ti.static_assert(isinstance(returned, tuple))
        first, second = returned
        return first.sum() + second.sum()

    assert result() == pytest.approx(15.0)


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_list_return_preserves_container(decorator):
    @decorator
    def values() -> tuple[ti.i32, ti.f32]:
        return [3.75, -2]

    @ti.kernel
    def result() -> ti.i32:
        returned = values()
        ti.static_assert(isinstance(returned, list))
        integer, real = returned
        return integer * 10 + ti.cast(real, ti.i32)

    assert result() == 28


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_single_scalar_return(decorator):
    @decorator
    def value() -> ti.i32:
        return 3.75

    @ti.kernel
    def result() -> ti.f32:
        return value() * 0.5

    assert result() == pytest.approx(1.5)


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_annotated_single_vector_return(decorator):
    @decorator
    def value() -> ti.math.vec2:
        return ti.Vector([2.0, 4.0])

    @ti.kernel
    def result() -> ti.math.vec2:
        return value() * 3

    assert result().to_list() == pytest.approx([6.0, 12.0])


@pytest.mark.parametrize("decorator", [ti.func, ti.pyfunc])
@test_utils.test()
def test_unannotated_tuple_return(decorator):
    @decorator
    def values():
        return 1.75, ti.Vector([2.0, 4.0])

    @ti.kernel
    def result() -> ti.f32:
        scalar, vector = values()
        return scalar + vector.sum()

    assert result() == pytest.approx(7.75)


def test_pyfunc_python_scope_tuple_return():
    @ti.pyfunc
    def values() -> tuple[ti.i32, ti.f32]:
        return 3.75, -2

    result = values()
    assert isinstance(result, tuple)
    assert result == (3.75, -2)
    assert isinstance(result[0], float)
    assert isinstance(result[1], int)
