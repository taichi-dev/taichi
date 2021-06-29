import pytest

import taichi as ti


def _test_trailing_bits():
    ti.init(arch=ti.cpu, debug=True, print_ir=True)

    x = ti.field(ti.f32)
    y = ti.field(ti.f32)

    block = ti.root.pointer(ti.i, 8)
    block.dense(ti.i, 32).place(x)

    # Here every 32 ti.i share the same dense node of 16 y along ti.j.
    block.dense(ti.j, 16).place(y)
    assert y.shape == (256, 16)
    # instead of y.shape == (8, 16),
    # since there are 5 trailing bits for ti.i for y's SNode

    assert x.shape == (256, )

    y[255, 15] = 0

    with pytest.raises(RuntimeError):
        y[256, 15] = 0

    with pytest.raises(RuntimeError):
        y[255, 16] = 0

    y[255, 3] = 123

    # They are the same element...
    assert y[255, 3] == 123
    assert y[254, 3] == 123
    assert y[240, 3] == 123


def _test_inconsistent_trailing_bits():
    ti.init(arch=ti.cpu, debug=True, print_ir=True)

    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)

    block = ti.root.pointer(ti.i, 8)

    # Here the numbers of bits of x and z are inconsistent,
    # which leads to the RuntimeError below.
    block.dense(ti.i, 32).place(x)
    block.dense(ti.i, 16).place(z)

    block.dense(ti.j, 16).place(y)

    with pytest.raises(RuntimeError):
        ti.get_runtime().materialize()
