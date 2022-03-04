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

    with pytest.raises(ti.TaichiSyntaxError, match="Type annotation can only be given to a single literal."):
        multi_args_error()


@test_utils.test()
def test_literal_keywords_error():
    @ti.kernel
    def keywords_error():
        a = ti.f64(1, x=2)

    with pytest.raises(ti.TaichiSyntaxError, match="Type annotation can only be given to a single literal."):
        keywords_error()


@test_utils.test()
def test_literal_expr_error():
    @ti.kernel
    def expr_error():
        a = 1
        b = ti.f16(a)

    with pytest.raises(ti.TaichiSyntaxError, match="Type annotation can only be given to a single literal."):
        expr_error()
