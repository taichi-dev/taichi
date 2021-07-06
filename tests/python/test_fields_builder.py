import pytest
from taichi.lang.exception import InvalidOperationError

import taichi as ti


@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_with_shape():
    n = 5
    x = ti.field(ti.f32, [n])

    @ti.kernel
    def func():
        for i in range(n):
            x[i] = i
        for i in range(n):
            assert x[i] == i

        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            assert x[i] == i * 2

    func()

    with pytest.raises(InvalidOperationError, match='FieldsBuilder finalized'):
        y = ti.field(ti.f32, [n])


@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder1():
    n = 5
    x = ti.field(ti.f32, [n])

    @ti.kernel
    def func1():
        for i in range(n):
            x[i] = i * 2

    func1()
    for i in range(n):
        assert x[i] == i * 2

    fb = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb.dense(ti.i, n).place(y)
    fb.finalize()

    @ti.kernel
    def func2():
        for i in range(n):
            y[i] = i // 2

    func2()
    for i in range(n):
        assert y[i] == i // 2

    func1()
    for i in range(n):
        assert x[i] == i * 2


@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder2():
    # TODO: x, y share the same memory location
    pass
    '''
    n = 5

    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.dense(ti.i, n).place(x)
    fb1.finalize()

    @ti.kernel
    def func1():
        for i in range(n):
            x[i] = i * 3

    func1()
    for i in range(n):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.dense(ti.i, n).place(y)
    fb2.finalize()

    @ti.kernel
    def func2():
        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            y[i] = i + 5

    func2()
    for i in range(n):
        assert x[i] == i * 2
        assert y[i] == i + 5
    '''
