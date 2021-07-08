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

    func()

    for i in range(n):
        assert x[i] == i

    y = ti.field(ti.f32, [n])

    @ti.kernel
    def func2():
        for i in range(n):
            y[i] = i * 2
        for i in range(n):
            x[i] = i * 3

    func2()

    for i in range(n):
        assert x[i] == i * 3
        assert y[i] == i * 2

    func()

    for i in range(n):
        assert x[i] == i


@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder_dense():
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
    z = ti.field(ti.f32)
    fb2.dense(ti.i, n).place(z)
    fb2.finalize()

    @ti.kernel
    def func2():
        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            y[i] = i + 5
        for i in range(n):
            z[i] = i + 10

    func2()
    for i in range(n):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    func1()
    for i in range(n):
        assert x[i] == i * 3


@ti.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder_pointer():
    n = 5

    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.pointer(ti.i, n).place(x)
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
    fb2.pointer(ti.i, n).place(y)
    z = ti.field(ti.f32)
    fb2.pointer(ti.i, n).place(z)
    fb2.finalize()

    @ti.kernel
    def func2():
        for i in range(n):
            x[i] = i * 2
        for i in range(n):
            y[i] = i + 5
        for i in range(n):
            z[i] = i + 10

    func2()
    for i in range(n):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    func1()
    for i in range(n):
        assert x[i] == i * 3
