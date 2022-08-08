import numpy as np
from pytest import approx

import taichi as ti
from tests import test_utils


@test_utils.test()
def test_struct_member_access():
    n = 32

    x = ti.Struct.field({"a": ti.f32, "b": ti.f32}, shape=(n, ))
    y = ti.Struct.field({"a": ti.f32, "b": ti.f32})

    ti.root.dense(ti.i, n // 4).dense(ti.i, 4).place(y)

    @ti.kernel
    def init():
        for i in x:
            x[i].a = i
            y[i].a = i

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            x[i].b = x[i].a

    def run_python_scope():
        for i in range(n):
            y[i].b = y[i].a * 2 + 1

    init()
    run_taichi_scope()
    for i in range(n):
        assert x[i].b == i
    run_python_scope()
    for i in range(n):
        assert y[i].b == i * 2 + 1


@test_utils.test()
def test_struct_whole_access():
    n = 32

    # also tests implicit cast
    x = ti.Struct.field({"a": ti.i32, "b": ti.f32}, shape=(n, ))
    y = ti.Struct.field({"a": ti.f32, "b": ti.i32})

    ti.root.dense(ti.i, n // 4).dense(ti.i, 4).place(y)

    @ti.kernel
    def init():
        for i in x:
            x[i] = ti.Struct(a=2 * i, b=1.01 * i)

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            # element-wise ops only work in Taichi scope
            y[i] = x[i] * 2 + 1

    def run_python_scope():
        for i in range(n):
            y[i] = ti.Struct(a=x[i].a, b=int(x[i].b))

    init()
    for i in range(n):
        assert x[i].a == 2 * i
        assert x[i].b == approx(1.01 * i, rel=1e-4)
    run_taichi_scope()
    for i in range(n):
        assert y[i].a == 4 * i + 1
        assert y[i].b == int((1.01 * i) * 2 + 1)
    run_python_scope()
    for i in range(n):
        assert y[i].a == 2 * i
        assert y[i].b == int(1.01 * i)


@test_utils.test()
def test_struct_fill():
    n = 32

    # also tests implicit cast
    x = ti.Struct.field({
        "a": ti.f32,
        "b": ti.types.vector(3, ti.i32)
    },
                        shape=(n, ))

    def fill_each():
        x.a.fill(1.0)
        x.b.fill(1.5)

    def fill_all():
        x.fill(2.5)

    @ti.kernel
    def fill_elements():
        for i in x:
            x[i].fill(i + 0.5)

    fill_each()
    for i in range(n):
        assert x[i].a == 1.0
        assert x[i].b[0] == 1 and x[i].b[1] == 1 and x[i].b[2] == 1
    fill_all()
    for i in range(n):
        assert x[i].a == 2.5
        assert x[i].b[0] == 2 and x[i].b[1] == 2 and x[i].b[2] == 2
    fill_elements()
    for i in range(n):
        assert x[i].a == i + 0.5
        assert np.allclose(x[i].b.to_numpy(), int(x[i].a))


@test_utils.test()
def test_matrix_type():
    n = 32
    vec2f = ti.types.vector(2, ti.f32)
    vec3i = ti.types.vector(3, ti.i32)
    x = vec3i.field()
    ti.root.dense(ti.i, n).place(x)

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            v = vec2f(i + 0.2)
            # also tests implicit cast
            x[i] = vec3i(v, i + 1.2)

    def run_python_scope():
        for i in range(n):
            v = vec2f(i + 0.2)
            x[i] = vec3i(i + 1.8, v)

    run_taichi_scope()
    for i in range(n):
        assert np.allclose(x[i].to_numpy(), np.array([i, i, i + 1]))
    run_python_scope()
    for i in range(n):
        assert np.allclose(x[i].to_numpy(), np.array([i + 1, i, i]))


@test_utils.test()
def test_struct_type():
    n = 32
    vec3f = ti.types.vector(3, float)
    line3f = ti.types.struct(linedir=vec3f, length=float)
    mystruct = ti.types.struct(line=line3f, idx=int)
    x = mystruct.field(shape=(n, ))

    @ti.kernel
    def init_taichi_scope():
        for i in x:
            x[i] = mystruct(1)

    def init_python_scope():
        for i in range(n):
            x[i] = mystruct(3)

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            v = vec3f(1)
            line = line3f(linedir=v, length=i + 0.5)
            x[i] = mystruct(line=line, idx=i)

    def run_python_scope():
        for i in range(n):
            v = vec3f(1)
            x[i] = ti.Struct({
                "line": {
                    "linedir": v,
                    "length": i + 0.5
                },
                "idx": i
            })

    init_taichi_scope()
    for i in range(n):
        assert x[i].idx == 0
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == 0.0
    run_taichi_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    init_python_scope()
    for i in range(n):
        assert x[i].idx == 0
        assert np.allclose(x[i].line.linedir.to_numpy(), 3.0)
        assert x[i].line.length == 0.0
    run_python_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    x.fill(5)
    for i in range(n):
        assert x[i].idx == 5
        assert np.allclose(x[i].line.linedir.to_numpy(), 5.0)
        assert x[i].line.length == 5.0


@test_utils.test(exclude=ti.cc)
def test_dataclass():
    # example struct class type
    vec3f = ti.types.vector(3, float)

    @ti.dataclass
    class Sphere:
        center: vec3f
        radius: ti.f32

        @ti.func
        def area(self):
            return 4 * 3.14 * self.radius * self.radius

        def py_scope_area(self):
            return 4 * 3.14 * self.radius * self.radius

    # test function usage from python scope
    assert np.isclose(
        Sphere(center=vec3f(0.0), radius=2.0).py_scope_area(),
        4.0 * 3.14 * 4.0)

    # test function usage from taichi scope
    @ti.kernel
    def get_area() -> ti.f32:
        sphere = Sphere(center=vec3f(0.0), radius=2.0)
        return sphere.area()

    assert np.isclose(get_area(), 4.0 * 3.14 * 4.0)

    # test function usage from taichi scope with field
    struct_field = Sphere.field(shape=(4, ))
    struct_field[3] = Sphere(center=vec3f(0.0), radius=2.0)

    @ti.kernel
    def get_area_field() -> ti.f32:
        return struct_field[3].area()

    assert np.isclose(get_area_field(), 4.0 * 3.14 * 4.0)


@test_utils.test()
def test_struct_assign():
    n = 32
    vec3f = ti.types.vector(3, float)
    line3f = ti.types.struct(linedir=vec3f, length=float)
    mystruct = ti.types.struct(line=line3f, idx=int)
    x = mystruct.field(shape=(n, ))
    y = line3f.field(shape=(n, ))

    @ti.kernel
    def init():
        for i in y:
            y[i] = line3f(linedir=vec3f(1), length=i + 0.5)

    @ti.kernel
    def run_taichi_scope():
        for i in x:
            x[i].idx = i
            x[i].line = y[i]

    def run_python_scope():
        for i in range(n):
            x[i].idx = i
            x[i].line = y[i]

    init()
    run_taichi_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    x.fill(5)
    run_python_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5


@test_utils.test()
def test_compound_type_implicit_cast():
    vec2i = ti.types.vector(2, int)
    vec2f = ti.types.vector(2, float)
    structi = ti.types.struct(a=int, b=vec2i)
    structf = ti.types.struct(a=float, b=vec2f)

    @ti.kernel
    def f2i_taichi_scope() -> int:
        s = structi(2.5, (2.5, 2.5))
        return s.a + s.b[0] + s.b[1]

    def f2i_python_scope():
        s = structi(2.5, (2.5, 2.5))
        return s.a + s.b[0] + s.b[1]

    @ti.kernel
    def i2f_taichi_scope() -> float:
        s = structf(2, (2, 2))
        return s.a + s.b[0] + s.b[1]

    def i2f_python_scope():
        s = structf(2, (2, 2))
        return s.a + s.b[0] + s.b[1]

    int_value = f2i_taichi_scope()
    assert type(int_value) == int and int_value == 6
    int_value = f2i_python_scope()
    assert type(int_value) == int and int_value == 6
    float_value = i2f_taichi_scope()
    assert type(float_value) == float and float_value == approx(6.0, rel=1e-4)
    float_value = i2f_python_scope()
    assert type(float_value) == float and float_value == approx(6.0, rel=1e-4)


@test_utils.test()
def test_local_struct_assign():
    n = 32
    vec3f = ti.types.vector(3, float)
    line3f = ti.types.struct(linedir=vec3f, length=float)
    mystruct = ti.types.struct(line=line3f, idx=int)

    @ti.kernel
    def run_taichi_scope():
        y = line3f(0)
        x = mystruct(0)
        x.idx = 0
        x.line = y

    def run_python_scope():
        y = line3f(0)
        x = mystruct(0)
        x.idx = 0
        x.line = y

    run_taichi_scope()
    run_python_scope()


@test_utils.test(debug=True)
def test_copy_python_scope_struct_to_taichi_scope():
    a = ti.Struct({'a': 2, 'b': 3})

    @ti.kernel
    def test():
        b = a
        assert b.a == 2
        assert b.b == 3
        b = ti.Struct({'a': 3, 'b': 4})
        assert b.a == 3
        assert b.b == 4

    test()


@test_utils.test(exclude=[ti.cc], debug=True)
def test_copy_struct_field_element_to_taichi_scope():
    a = ti.Struct.field({'a': ti.i32, 'b': ti.i32}, shape=())
    a[None].a = 2
    a[None].b = 3

    @ti.kernel
    def test():
        b = a[None]
        assert b.a == 2
        assert b.b == 3
        b.a = 5
        b.b = 9
        assert b.a == 5
        assert b.b == 9
        assert a[None].a == 2
        assert a[None].b == 3

    test()


@test_utils.test(debug=True)
def test_copy_struct_in_taichi_scope():
    @ti.kernel
    def test():
        a = ti.Struct({'a': 2, 'b': 3})
        b = a
        assert b.a == 2
        assert b.b == 3
        b.a = 5
        b.b = 9
        assert b.a == 5
        assert b.b == 9
        assert a.a == 2
        assert a.b == 3

    test()


@test_utils.test(debug=True)
def test_dataclass():
    vec3 = ti.types.vector(3, float)

    @ti.dataclass
    class Foo:
        pos: vec3
        vel: vec3
        mass: float

    @ti.kernel
    def test():
        A = Foo((1, 1, 1), mass=2)
        assert all(A.pos == [1.0, 1.0, 1.0])
        assert all(A.vel == [0.0, 0.0, 0.0])
        assert A.mass == 2.0

    test()
