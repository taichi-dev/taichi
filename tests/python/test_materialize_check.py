import pytest

import taichi as ti


@ti.test()
def test_check_field_not_placed():
    a = ti.field(ti.i32)

    @ti.kernel
    def foo():
        pass

    with pytest.raises(RuntimeError,
                       match=r"These field\(s\) are not placed.*"):
        foo()


@ti.test()
def test_check_matrix_field_member_shape():
    a = ti.Matrix.field(2, 2, ti.i32)
    ti.root.dense(ti.i, 10).place(a.get_scalar_field(0, 0))
    ti.root.dense(ti.i, 11).place(a.get_scalar_field(0, 1))
    ti.root.dense(ti.i, 10).place(a.get_scalar_field(1, 0))
    ti.root.dense(ti.i, 11).place(a.get_scalar_field(1, 1))

    @ti.kernel
    def foo():
        pass

    with pytest.raises(
            RuntimeError,
            match=r"Members of the following field have different shapes.*"):
        foo()
