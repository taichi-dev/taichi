import pytest

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_literal_u32():
    @ti.kernel
    def pcg_hash(inp: ti.u32) -> ti.u32:
        state: ti.u32 = inp * ti.u32(747796405) + ti.u32(2891336453)
        word: ti.u32 = ((state >> (
            (state >> ti.u32(28)) + ti.u32(4))) ^ state) * ti.u32(277803737)
        return (word >> ti.u32(22)) ^ word

    assert pcg_hash(12345678) == 119515934
    assert pcg_hash(98765432) == 4244201195


@test_utils.test()
def test_literal_multi_args_error():
    @ti.kernel
    def multi_args_error():
        a = ti.i64(1, 2)

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="A primitive type can only decorate a single expression."):
        multi_args_error()


@test_utils.test()
def test_literal_keywords_error():
    @ti.kernel
    def keywords_error():
        a = ti.f64(1, x=2)

    with pytest.raises(
            ti.TaichiSyntaxError,
            match="A primitive type can only decorate a single expression."):
        keywords_error()


@test_utils.test()
def test_literal_compound_error():
    @ti.kernel
    def expr_error():
        a = ti.Vector([1])
        b = ti.f16(a)

    with pytest.raises(
            ti.TaichiSyntaxError,
            match=
            "A primitive type cannot decorate an expression with a compound type."
    ):
        expr_error()


@test_utils.test()
def test_literal_int_annotation_error():
    @ti.kernel
    def int_annotation_error():
        a = ti.f32(0)

    with pytest.raises(
            ti.TaichiTypeError,
            match=
            "Integer literals must be annotated with a integer type. For type casting, use `ti.cast`."
    ):
        int_annotation_error()


@test_utils.test()
def test_literal_float_annotation_error():
    @ti.kernel
    def float_annotation_error():
        a = ti.i32(0.0)

    with pytest.raises(
            ti.TaichiTypeError,
            match=
            "Floating-point literals must be annotated with a floating-point type. For type casting, use `ti.cast`."
    ):
        float_annotation_error()


@test_utils.test()
def test_literal_exceed_default_ip():
    @ti.kernel
    def func():
        b = 0x80000000

    with pytest.raises(ti.TaichiTypeError,
                       match="exceeded the range of default_ip"):
        func()


@test_utils.test()
def test_literal_exceed_specified_dtype():
    @ti.kernel
    def func():
        b = ti.u16(-1)

    with pytest.raises(ti.TaichiTypeError,
                       match="exceeded the range of specified dtype"):
        func()
