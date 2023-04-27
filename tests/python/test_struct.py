import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize("round", range(10))
@test_utils.test()
def test_linear(round):
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n).place(x)
    ti.root.dense(ti.i, n).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x)
    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested_aos():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.i, n // 16).dense(ti.i, 16).place(x, y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test(exclude=[ti.vulkan, ti.dx11])
def test_2d_nested():
    x = ti.field(ti.i32)

    n = 128

    ti.root.dense(ti.ij, n // 16).dense(ti.ij, (32, 16)).place(x)

    for i in range(n * 2):
        for j in range(n):
            x[i, j] = i + j * 10

    for i in range(n * 2):
        for j in range(n):
            assert x[i, j] == i + j * 10


@test_utils.test()
def test_func_of_data_class_as_kernel_arg():
    @ti.dataclass
    class Foo:
        x: ti.f32
        y: ti.f32

        @ti.func
        def add(self, other: ti.template()):
            return Foo(self.x + other.x, self.y + other.y)

    @ti.kernel
    def foo_x(x: Foo) -> ti.f32:
        return x.add(x).x

    assert foo_x(Foo(1, 2)) == 2

    @ti.kernel
    def foo_y(x: Foo) -> ti.f32:
        return x.add(x).y

    assert foo_y(Foo(1, 2)) == 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.amdgpu])
def test_func_of_data_class_as_kernel_return():
    # TODO: enable this test in SPIR-V based backends after SPIR-V based backends can return structs.
    @ti.dataclass
    class Foo:
        x: ti.f32
        y: ti.f32

        @ti.func
        def add(self, other: ti.template()):
            return Foo(self.x + other.x, self.y + other.y)

        def add_python(self, other):
            return Foo(self.x + other.x, self.y + other.y)

    @ti.kernel
    def foo(x: Foo) -> Foo:
        return x.add(x)

    b = foo(Foo(1, 2))
    assert b.x == 2
    assert b.y == 4

    c = b.add_python(b)
    assert c.x == 4
    assert c.y == 8


@test_utils.test()
def test_nested_data_class_func():
    @ti.dataclass
    class Foo:
        a: int

        @ti.func
        def foo(self):
            return self.a

    @ti.dataclass
    class Nested:
        f: Foo

        @ti.func
        def testme(self) -> int:
            return self.f.foo()

    @ti.kernel
    def k() -> int:
        x = Nested(Foo(42))
        return x.testme()

    assert k() == 42
